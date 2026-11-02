"""Demonstration-conditioned object assistance, inspired by DexMachina.

The helper changes simulator forces, not the robot action space. Reference
states and contact labels are privileged training information, never prompts.
This is a Panda adaptation, not a reproduction of the dexterous-hand paper.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

import numpy as np
from scipy.spatial.transform import Rotation

from .sim.embodiment import Start


@dataclass
class AssistanceConfig:
    gain: float = 1.0
    kp: float = 30.0                 # N/m before gain scaling
    rotation_kp: float = .15         # Nm/radian
    max_force: float = 3.0           # per-object vector norm, N
    max_torque: float = .10          # per-object vector norm, Nm
    object_weight: float = .3
    motion_weight: float = .2
    contact_weight: float = .5

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v >= 0 for v in values.values()):
            raise ValueError('Assistance parameters must be finite and nonnegative')
        if self.gain > 1 or self.kp <= 0 or self.rotation_kp <= 0:
            raise ValueError('Gain must be in [0,1]; PD stiffnesses must be positive')


def bounded(vector, limit):
    norm = np.linalg.norm(vector)
    return vector * min(1., limit / max(float(norm), 1e-12))


def wrench(position, rotation, velocity, angular_velocity, target_position, target_rotation,
           mass, inertia, cfg):
    """World-frame force/torque with mass-aware damping and norm caps.

    Scaling spring by gain and damping by sqrt(gain) keeps critical damping
    for the translational approximation. Both vanish exactly at gain=0.
    """
    if cfg.gain == 0:
        return np.zeros(6)
    spring = cfg.gain * cfg.kp
    angular_spring = cfg.gain * cfg.rotation_kp
    force = spring * (target_position - position) - 2 * np.sqrt(spring * mass) * velocity
    error = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
    torque = angular_spring * error - 2 * np.sqrt(angular_spring * np.max(inertia)) * angular_velocity
    return np.r_[bounded(force, cfg.gain * cfg.max_force),
                 bounded(torque, cfg.gain * cfg.max_torque)]


def load_reference(path):
    with np.load(path, allow_pickle=False) as source:
        ref = {key: source[key].copy() for key in source.files}
    names = ref['body_names'].tolist()
    t, n = ref['object_pos'].shape[:2]
    shapes = dict(object_pos=(t, n, 3), object_mat=(t, n, 3, 3),
                  eef_pos=(t, 3), eef_mat=(t, 3, 3), grasp=(t, n))
    if t < 2 or n < 1 or len(set(names)) != n:
        raise ValueError('Reference needs distinct objects and at least two frames')
    for key, shape in shapes.items():
        if ref[key].shape != shape or not np.all(np.isfinite(ref[key])):
            raise ValueError(f'Invalid reference array {key}')
    if not np.all(np.isfinite(ref['initial_state'])) or ref['initial_state'].ndim != 1:
        raise ValueError('Invalid initial simulator state')
    if not np.any(ref['grasp']):
        raise ValueError('Reference has no verified grasp frames')
    for key in ('object_mat', 'eef_mat'):
        matrices = ref[key].reshape(-1, 3, 3)
        if not np.allclose(matrices @ matrices.transpose(0, 2, 1), np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(matrices), 1., atol=1e-4):
            raise ValueError('Reference orientations must be proper rotation matrices')
    return ref


class AssistanceCurriculum:
    """Decay only after motion, contact AND object tracking meet thresholds.

    A new level requires a fresh window. An episode retains its reset gain;
    outcomes from older gain levels cannot trigger further reductions.
    """
    def __init__(self, gain=1., window=5, decay=.8, minimum=.05,
                 task_threshold=.8, motion_threshold=.6, contact_threshold=.5, success_threshold=.8):
        if not 0 <= gain <= 1 or window < 1 or not 0 < decay < 1 or not 0 < minimum < 1:
            raise ValueError('Invalid assistance curriculum parameters')
        if not all(0 <= v <= 1 for v in (task_threshold, motion_threshold, contact_threshold, success_threshold)):
            raise ValueError('Tracking thresholds must be in [0,1]')
        self.gain, self.window, self.decay, self.minimum = gain, window, decay, minimum
        self.thresholds = dict(object_score=task_threshold, motion_score=motion_threshold,
                               contact_score=contact_threshold, success=success_threshold)
        self.history = []

    def observe(self, result):
        if self.gain == 0 or not np.isclose(result['gain'], self.gain):
            return
        if not all(np.isfinite(result[k]) and 0 <= result[k] <= 1 + 1e-6 for k in self.thresholds):
            raise ValueError('Invalid curriculum tracking scores')
        self.history = (self.history + [{k: float(result[k]) for k in self.thresholds}])[-self.window:]

    def advance(self):
        if self.gain == 0 or len(self.history) < self.window:
            return False
        if not all(np.mean([r[k] for r in self.history]) >= v for k, v in self.thresholds.items()):
            return False
        candidate = self.gain * self.decay
        self.gain = 0. if candidate < self.minimum else candidate
        self.history = []
        return True

    def state_dict(self):
        return dict(gain=self.gain, history=self.history, window=self.window, decay=self.decay,
                    minimum=self.minimum, thresholds=self.thresholds)

    def load_state_dict(self, state):
        for key in ('window', 'decay', 'minimum', 'thresholds'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Cannot change curriculum {key} when resuming')
        gain = float(state['gain'])
        history = state['history']
        if not 0 <= gain <= 1 or len(history) > self.window:
            raise ValueError('Invalid saved curriculum state')
        self.gain, self.history = gain, history


class AssistedStartSampler:
    """Full demonstration starts with their own matched reference trajectories."""
    def __init__(self, folder, task, config=None, curriculum=None, seed=0):
        self.folder = Path(folder).resolve()
        manifest_text = (self.folder / 'manifest.json').read_text()
        self.manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
        manifest = json.loads(manifest_text)
        if manifest['suite'] != 'libero_10' or manifest['task_id'] != task:
            raise ValueError('Assistance references do not match the training task/suite')
        self.config = config or AssistanceConfig()
        self.curriculum = curriculum or AssistanceCurriculum(gain=self.config.gain)
        self.rng = random.Random(seed)
        self.references = []
        for item in manifest['references']:
            path = (self.folder / item['file']).resolve()
            if not path.is_relative_to(self.folder):
                raise ValueError('Reference path leaves the dataset directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Reference checksum mismatch')
            ref = load_reference(path)
            if int(ref['task_id']) != task:
                raise ValueError('Reference task mismatch')
            self.references.append((path, ref))
        if not self.references:
            raise ValueError('No assistance references')

    def __call__(self):
        return self.start_at(self.rng.randrange(len(self.references)))

    def start_at(self, index):
        path, ref = self.references[index]
        return Start('libero_10', int(ref['task_id']), sim_state=ref['initial_state'],
                     label=f'voc_demo{int(ref["demo_id"])}_from_start',
                     settle_gripper=float(ref.get('settle_gripper', -1.)),
                     assistance=asdict(self.config) | dict(reference=str(path), gain=self.curriculum.gain))

    def state_dict(self):
        return dict(manifest_hash=self.manifest_hash, curriculum=self.curriculum.state_dict())

    def load_state_dict(self, state):
        if state['manifest_hash'] != self.manifest_hash:
            raise ValueError('Assistance reference dataset changed since checkpoint')
        self.curriculum.load_state_dict(state['curriculum'])


class ObjectAssistant:
    """PD force helper installed at every MuJoCo physics substep.

    Only target object xfrc_applied rows are written. No teleportation, changes
    to mass/gravity, robot actuation or observation text are performed.
    """
    def __init__(self, domain, reference, config):
        self.domain, self.ref, self.cfg = domain, load_reference(reference), config
        self.names = self.ref['body_names'].tolist()
        self.ids = [domain.obj_body_id[name] for name in self.names]
        targets = {str(g[1]) for g in domain.parsed_problem['goal_state'] if len(g) == 3}
        if set(self.names) != targets:
            raise ValueError('Assistance may act only on this task\'s target objects')
        self.body_names = [domain.sim.model.body_id2name(i) for i in self.ids]
        data = domain.sim.data
        initial_pos = data.body_xpos[self.ids]
        if np.max(np.linalg.norm(initial_pos - self.ref['object_pos'][0], axis=-1)) > .03:
            raise ValueError('Reference does not match initial object arrangement (3 cm tolerance)')
        if np.linalg.norm(data.site_xpos[domain.robots[0].eef_site_id] - self.ref['eef_pos'][0]) > .05:
            raise ValueError('Reference does not match initial robot pose (5 cm tolerance)')
        self.baseline = data.xfrc_applied[self.ids].copy()
        self.original_pre_action = domain._pre_action
        self.frame = 0
        self.steps = 0
        self.force_squared = 0.
        self.impulse = 0.
        self.physics_samples = 0
        self.max_force_seen = 0.
        self.torque_squared = 0.
        self.max_torque_seen = 0.
        self.sums = dict(object=0., motion=0., contact=0.)
        self.contact_frames = max(1, int(np.any(self.ref['grasp'][1:], axis=1).sum()))
        self.length = len(self.ref['eef_pos']) - 1

        def pre_action(action, policy_step=False):
            self.original_pre_action(action, policy_step)
            self.apply()
        domain._pre_action = pre_action

    def apply(self):
        domain, data, model = self.domain, self.domain.sim.data, self.domain.sim.model
        data.xfrc_applied[self.ids] = self.baseline
        if self.frame >= self.length or self.cfg.gain == 0:
            return
        target_frame = self.frame + 1
        for index, (body, name) in enumerate(zip(self.ids, self.body_names)):
            force = wrench(data.body_xpos[body], data.body_xmat[body].reshape(3, 3),
                           data.get_body_xvelp(name), data.get_body_xvelr(name),
                           self.ref['object_pos'][target_frame, index], self.ref['object_mat'][target_frame, index],
                           model.body_mass[body], model.body_inertia[body], self.cfg)
            data.xfrc_applied[body] += force
            norm = float(np.linalg.norm(force[:3]))
            self.max_force_seen = max(self.max_force_seen, norm)
            torque_norm = float(np.linalg.norm(force[3:]))
            self.max_torque_seen = max(self.max_torque_seen, torque_norm)
            self.torque_squared += torque_norm * torque_norm
            self.force_squared += norm * norm
            self.impulse += norm * domain.model_timestep
            self.physics_samples += 1

    def before_step(self, frame):
        self.frame = frame

    def after_step(self, grasped):
        if self.frame >= self.length:
            return
        i = self.frame + 1
        data = self.domain.sim.data
        errors = np.linalg.norm(data.body_xpos[self.ids] - self.ref['object_pos'][i], axis=-1)
        current_mat = data.body_xmat[self.ids].reshape(-1, 3, 3)
        angles = Rotation.from_matrix(self.ref['object_mat'][i] @ current_mat.transpose(0, 2, 1)).magnitude()
        self.sums['object'] += float(np.mean(np.exp(-20 * errors - angles)))
        hand = data.site_xpos[self.domain.robots[0].eef_site_id]
        hand_error = np.linalg.norm(hand - self.ref['eef_pos'][i])
        self.sums['motion'] += float(np.exp(-20 * hand_error))
        expected = [name for name, yes in zip(self.names, self.ref['grasp'][i]) if yes]
        if expected:
            self.sums['contact'] += sum(name in grasped for name in expected) / len(expected)
        self.steps += 1

    def summary(self):
        # The first reference state is used as an initial state; reference frames
        # past the trajectory end earn no further imitation reward.
        object_score = min(1., self.sums['object'] / self.length)
        motion_score = min(1., self.sums['motion'] / self.length)
        contact_score = min(1., self.sums['contact'] / self.contact_frames)
        returns = dict(voc_object=self.cfg.object_weight * object_score,
                       voc_motion=self.cfg.motion_weight * motion_score,
                       voc_contact=self.cfg.contact_weight * contact_score)
        return dict(gain=self.cfg.gain, assisted=self.cfg.gain > 0, reference_frames=self.length,
                    reference_frame=min(self.frame, self.length), object_score=object_score,
                    motion_score=motion_score, contact_score=contact_score, returns=returns,
                    force_rms_N=float(np.sqrt(self.force_squared / max(1, self.physics_samples))),
                    max_force_N=self.max_force_seen,
                    torque_rms_Nm=float(np.sqrt(self.torque_squared / max(1, self.physics_samples))),
                    max_torque_Nm=self.max_torque_seen,
                    impulse_Ns=self.impulse)

    def close(self):
        self.domain._pre_action = self.original_pre_action
        self.domain.sim.data.xfrc_applied[self.ids] = self.baseline
