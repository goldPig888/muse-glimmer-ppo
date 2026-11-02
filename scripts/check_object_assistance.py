"""Physics-only check of object assistance, including a zero-force control.

The robot holds position unless --follow-hand is supplied. Success with an
idle robot is helper performance, not learned robot performance. No GPU model
or optimizer is loaded. Frames are saved under the output's episodes/.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from glimmer_ppo.assistance import AssistedStartSampler, AssistanceConfig
from glimmer_ppo.paper_env import PaperEnv, PaperEnvConfig
from glimmer_ppo.reward import PotentialConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object-references', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', type=int, default=0)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--follow-hand', action='store_true', help='follow recorded end-effector poses; not a learned policy')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error('Use a fresh output directory')
    if args.steps < 1:
        p.error('--steps must be positive')
    sampler = AssistedStartSampler(args.object_references, args.task)
    path, ref = sampler.references[0]
    # Use exactly the same first reference and simulator seed for both runs.
    template = sampler.start_at(0)
    template = replace(template, sim_state=ref['initial_state'], label='physics_check')
    rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for gain in (1., 0.):
        from dataclasses import asdict
        settings = asdict(AssistanceConfig(gain=gain)) | dict(reference=str(path))
        start = replace(template, assistance=settings, label=f'physics_gain{gain:g}')
        env = PaperEnv(args.output / 'episodes', PaperEnvConfig(max_calls=args.steps,
            max_steps=args.steps, reward=PotentialConfig(mode='sparse')))
        try:
            env.reset(start)
            done, info = False, {}
            while not done:
                if args.follow_hand:
                    i = min(env.episode.steps + 5, len(ref['eef_pos']) - 1)
                    xyz = ref['eef_pos'][i]
                    matrix = ref['eef_mat'][i, :, :2].T.reshape(-1)
                    targets = dict(zip(('x', 'y', 'z', 'r00', 'r10', 'r20', 'r01', 'r11', 'r21'),
                                       [*xyz, *matrix]))
                    targets['gripper'] = float(ref['demo_actions'][min(i, len(ref['demo_actions']) - 1), -1] > 0)
                else:
                    targets = {'gripper': 0.}
                _, done, info = env.step(json.dumps(dict(name='move_to', arguments=dict(
                    targets=targets, note='Scripted physics check, not a learned action.'))))
            result = dict(info['voc'], robot='reference_hand' if args.follow_hand else 'idle',
                          success=info['success'], steps=info['steps'])
            if gain == 0 and (result['impulse_Ns'] != 0 or result['force_rms_N'] != 0 or result['torque_rms_Nm'] != 0):
                raise RuntimeError('Zero-assistance control applied an external force')
            if result['max_force_N'] > 3. + 1e-6:
                raise RuntimeError('Assistance exceeded configured force cap')
            if result['max_torque_Nm'] > .1 + 1e-6:
                raise RuntimeError('Assistance exceeded configured torque cap')
            rows.append(result)
            print(json.dumps(result), flush=True)
        finally:
            env.close()
    (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('Physics check complete. Assisted/scripted success is not policy evaluation.', flush=True)


if __name__ == '__main__':
    main()
