import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from glimmer_ppo.assistance import (AssistanceConfig, AssistanceCurriculum, AssistedStartSampler,
                                    ObjectAssistant, load_reference, wrench)


def reference(path):
    arrays = dict(task_id=0, demo_id=10, body_names=np.array(['can']), initial_state=np.array([1., 2.]),
                  eef_pos=np.zeros((4, 3)), eef_mat=np.tile(np.eye(3), (4, 1, 1)),
                  object_pos=np.zeros((4, 1, 3)), object_mat=np.tile(np.eye(3), (4, 1, 1, 1)),
                  grasp=np.array([[False], [True], [True], [False]]))
    arrays['object_pos'][1:, 0, 2] = .01
    np.savez(path, **arrays)
    return arrays


def test_pd_force_is_bounded_and_zero_gain_is_exactly_zero():
    args = (np.zeros(3), np.eye(3), np.array([1., 2., 3.]), np.ones(3),
            np.array([100., 100., 100.]), np.diag([-1., -1., 1.]), .1, np.array([.01, .01, .01]))
    for gain in (0., .1, 1.):
        cfg = AssistanceConfig(gain=gain)
        force = wrench(*args, cfg)
        assert np.linalg.norm(force[:3]) <= gain * cfg.max_force + 1e-12
        assert np.linalg.norm(force[3:]) <= gain * cfg.max_torque + 1e-12
        if gain == 0:
            np.testing.assert_array_equal(force, np.zeros(6))
    for bad in (dict(gain=-1), dict(gain=float('nan')), dict(kp=0), dict(max_force=-1)):
        with pytest.raises(ValueError):
            AssistanceConfig(**bad)


def test_curriculum_requires_robot_participation_and_restores_history():
    curriculum = AssistanceCurriculum(window=2, decay=.5, minimum=.3)
    # The helper can carry the object while the robot does nothing: this must
    # NOT trigger the hand-off merely because object tracking is perfect.
    for _ in range(2):
        curriculum.observe(dict(gain=1., object_score=1., motion_score=0., contact_score=0., success=True))
    assert not curriculum.advance()
    good = dict(gain=1., object_score=.9, motion_score=.9, contact_score=.9, success=True)
    curriculum.observe(good)
    restored = AssistanceCurriculum(window=2, decay=.5, minimum=.3)
    restored.load_state_dict(curriculum.state_dict())
    assert not restored.advance()
    restored.observe(good)
    assert restored.advance() and restored.gain == .5
    assert restored.history == []
    restored.observe(good)  # stale episode at the preceding gain is ignored
    assert restored.history == []
    for _ in range(2):
        restored.observe(dict(good, gain=.5))
    assert restored.advance() and restored.gain == 0.


def test_reference_sampler_starts_at_frame_zero_and_detects_dataset_changes(tmp_path):
    path = tmp_path / 'demo.npz'
    reference(path)
    manifest = dict(suite='libero_10', task_id=0, references=[dict(file=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    sampler = AssistedStartSampler(tmp_path, 0)
    start = sampler()
    np.testing.assert_array_equal(start.sim_state, [1., 2.])
    assert start.label == 'voc_demo10_from_start'
    assert start.assistance['gain'] == 1.
    sampler.curriculum.gain = .2
    saved = sampler.state_dict()
    restored = AssistedStartSampler(tmp_path, 0)
    restored.load_state_dict(saved)
    assert restored().assistance['gain'] == .2
    bad = dict(saved, manifest_hash='wrong')
    with pytest.raises(ValueError, match='changed'):
        restored.load_state_dict(bad)
    with pytest.raises(ValueError, match='task/suite'):
        AssistedStartSampler(tmp_path, 1)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        AssistedStartSampler(tmp_path, 0)


def test_force_hook_only_changes_target_object_never_accumulates_and_clears(tmp_path):
    path = tmp_path / 'demo.npz'
    reference(path)
    data = SimpleNamespace(body_xpos=np.zeros((2, 3)), body_xmat=np.tile(np.eye(3).reshape(9), (2, 1)),
                           site_xpos=np.zeros((1, 3)), xfrc_applied=np.array([[7.] * 6, [0.] * 6]),
                           get_body_xvelp=lambda name: np.zeros(3), get_body_xvelr=lambda name: np.zeros(3))
    model = SimpleNamespace(body_id2name=lambda i: 'can_body', body_mass=np.array([1., .1]),
                            body_inertia=np.full((2, 3), .01))
    calls = []
    original = lambda a, p: calls.append((a, p))
    domain = SimpleNamespace(sim=SimpleNamespace(data=data, model=model), obj_body_id={'can': 1},
                             robots=[SimpleNamespace(eef_site_id=0)], model_timestep=.002,
                             parsed_problem={'goal_state': [['In', 'can', 'basket']]}, _pre_action=original)
    helper = ObjectAssistant(domain, path, AssistanceConfig())
    helper.before_step(0)
    domain._pre_action('action', True)
    first = data.xfrc_applied.copy()
    domain._pre_action('action', False)
    np.testing.assert_array_equal(first, data.xfrc_applied)
    np.testing.assert_array_equal(first[0], [7.] * 6)
    assert first[1, 2] > 0
    helper.after_step([])
    assert helper.summary()['contact_score'] == 0
    assert helper.summary()['impulse_Ns'] > 0
    # Out-of-reference frames earn no further imitation reward or helper force.
    helper.before_step(4)
    helper.apply()
    before = helper.summary()['returns']
    helper.after_step(['can'])
    assert helper.summary()['returns'] == before
    np.testing.assert_array_equal(data.xfrc_applied[1], np.zeros(6))
    helper.close()
    assert domain._pre_action is original
    assert len(calls) == 2


def test_reference_rejects_invalid_rotations(tmp_path):
    path = tmp_path / 'demo.npz'
    arrays = reference(path)
    arrays['eef_mat'][1] = 0
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='rotation'):
        load_reference(path)


def test_assistance_rewards_are_integrated_once_and_not_exposed_in_messages(tmp_path):
    from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
    from glimmer_ppo.reward import PotentialConfig
    env = PaperEnv.__new__(PaperEnv)
    env.cfg = PaperEnvConfig(reward=PotentialConfig(mode='sparse'))
    env.progress = {}
    env.assistance_returns = {}
    env.failed_calls = set()
    env.messages = []
    summary = dict(gain=.5, returns={'voc_object': .2, 'voc_motion': .1, 'voc_contact': .3})
    env.embodiment = SimpleNamespace(progress=dict(voc=summary))
    episode = SimpleNamespace(success=False, over=False, outcome=None, steps=1, calls=0, run_dir=tmp_path)
    def call(name, args):
        episode.calls += 1
        return SimpleNamespace(text='motion complete', frames=[])
    episode.call = call
    env.episode = episode
    first, _, info = env.step('{"name":"move_to","arguments":{"targets":{"z":0.6}}}')
    second, _, _ = env.step('{"name":"move_to","arguments":{"targets":{"z":0.6}}}')
    assert first == pytest.approx(.6 - .005)
    assert second == pytest.approx(-.005)
    assert info['voc']['gain'] == .5
    assert 'voc' not in json.dumps(env.messages)
    # Avoid PaperEnv.close touching this fake after the test.
    env.episode = None


def test_frozen_reference_evaluation_defaults_to_zero_force(tmp_path, monkeypatch):
    from pathlib import Path
    import runpy
    import sys
    import torch
    path = tmp_path / 'demo.npz'
    reference(path)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(suite='libero_10', task_id=0,
        references=[dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())])))
    script = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/eval_task.py'))
    main = script['main']
    gains = []
    class Policy(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()

        def act(self, messages, tools):
            assert not self.training and not torch.is_grad_enabled()
            return 'move', False, {}
    class Env:
        tools = None
        messages = []

        def __init__(self, *args):
            pass

        def reset(self, start):
            gains.append(start.assistance['gain'])
            np.testing.assert_array_equal(start.sim_state, [1., 2.])
            return []

        def step(self, *args):
            return 0., True, dict(success=True, outcome='success', decisions=1, steps=1,
                goal_fraction=1., grasped=[], lifted=[], voc=dict(gain=0., max_force_N=0., max_torque_Nm=0.))

        def close(self):
            pass
    monkeypatch.setitem(main.__globals__, 'Policy', Policy)
    monkeypatch.setitem(main.__globals__, 'PaperEnv', Env)
    out = tmp_path / 'evaluation'
    monkeypatch.setattr(sys, 'argv', ['eval_task', '--object-references', str(tmp_path),
                                    '--output', str(out), '--samples', '2'])
    main()
    assert gains == [0., 0.]
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['assistance_gain'] == 0.
    assert summary['episodes'] == 2


def test_reference_export_uses_first_state_and_verified_grasp_frames(tmp_path, monkeypatch):
    import h5py
    from pathlib import Path
    import runpy
    import sys
    from types import ModuleType
    states = np.array([[0., 99.], [1., 99.], [2., 99.]])
    with h5py.File(tmp_path / 'task_demo.hdf5', 'w') as data:
        group = data.create_group('data/demo_10')
        group['states'] = states
        group['actions'] = np.zeros((3, 7))
    class Env:
        frame = 0

        def __init__(self, **kwargs):
            data = SimpleNamespace(body_xpos=np.zeros((1, 3)), body_xmat=np.eye(3).reshape(1, 9),
                                   site_xpos=np.zeros((1, 3)), site_xmat=np.eye(3).reshape(1, 9))
            self.env = SimpleNamespace(parsed_problem={'goal_state': [['In', 'can', 'basket']]},
                obj_body_id={'can': 0}, objects_dict={'can': object()}, robots=[SimpleNamespace(eef_site_id=0, gripper=object())],
                sim=SimpleNamespace(data=data, forward=lambda: None),
                _check_grasp=lambda **kwargs: self.frame == 1,
                _eval_predicate=lambda goal: self.frame == 2)

        def seed(self, seed):
            pass

        def reset(self):
            self.frame = 0

        def set_init_state(self, state):
            self.set_state(state)

        def set_state(self, state):
            self.frame = int(state[0])
            self.env.sim.data.body_xpos[:] = self.frame

        def check_success(self):
            return self.frame == 2

        def close(self):
            pass
    task = SimpleNamespace(name='task', problem_folder='libero_10', bddl_file='task.bddl')
    suite = SimpleNamespace(get_task=lambda index: task)
    module = ModuleType('libero.libero')
    module.benchmark = SimpleNamespace(get_benchmark_dict=lambda: {'libero_10': lambda: suite})
    module.get_libero_path = lambda name: str(tmp_path)
    env_module = ModuleType('libero.libero.envs')
    env_module.OffScreenRenderEnv = Env
    monkeypatch.setitem(sys.modules, 'libero.libero', module)
    monkeypatch.setitem(sys.modules, 'libero.libero.envs', env_module)
    main = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/prepare_object_references.py'))['main']
    out = tmp_path / 'references'
    monkeypatch.setattr(sys, 'argv', ['prepare', '--demos', str(tmp_path), '--output', str(out), '--demo-ids', '10'])
    main()
    exported = load_reference(out / 'demo_0010.npz')
    np.testing.assert_array_equal(exported['initial_state'], states[0])
    assert exported['grasp'].tolist() == [[False], [True], [False]]
    assert json.loads((out / 'manifest.json').read_text())['start_kind'] == 'full_demo_initial_state'
