import json
from pathlib import Path
import runpy

module = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/serve_training_viewer.py'))
RunReader = module['RunReader']


def test_jsonl_retries_partial_records_and_handles_replacement(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(b'{"update":1}\n{"update":')
    reader = RunReader(tmp_path)
    assert reader.rows(path) == [{'update': 1}]
    with path.open('ab') as f:
        f.write(b'2}\n')
    assert reader.rows(path) == [{'update': 1}, {'update': 2}]
    assert reader.rows(path) == [{'update': 1}, {'update': 2}]
    path.write_text('{"update":3}\n')
    assert reader.rows(path) == [{'update': 3}]


def test_snapshot_before_first_update_and_after_completion(tmp_path):
    (tmp_path / 'config.json').write_text(json.dumps({'args': {
        'updates': '1', 'decisions_per_update': '2',
    }}))
    ep = tmp_path / 'episodes/episode_00000_task00_init36'
    (ep / 'frames').mkdir(parents=True)
    (ep / 'protocol.json').write_text('{"init_state":36}')
    (ep / 'ppo_turns.jsonl').write_text('{"decision":1,"reward":0.1}\n')
    (ep / 'frames/call09_agentview.png').touch()
    (ep / 'frames/call100_agentview.png').touch()
    reader = RunReader(tmp_path)
    reader.gpu = lambda: []
    initial = reader.snapshot('../../unrelated')
    assert initial['total_decisions'] == 1
    assert initial['batch_collected'] == 1
    assert initial['phase'] == 'stopped'
    assert initial['protocol']['init_state'] == 36
    assert initial['frames']['agentview']['path'].endswith('call100_agentview.png')
    (tmp_path / 'metrics.jsonl').write_text('{"update":1,"decisions":2,"success_rate":NaN}\n')
    (tmp_path / 'episodes.jsonl').write_text('{"success":true}\n')
    final = reader.snapshot()
    assert final['phase'] == 'complete'
    assert final['metrics'][0]['success_rate'] is None
    assert final['successes'] == 1
    json.dumps(final, allow_nan=False)


def test_optimizer_curves_follow_incremental_tqdm_output(tmp_path):
    log = tmp_path / 'train.log'
    log.write_bytes(b'[ppo] update 3: optimizing (critic warmup)\n'
                    b'\rPPO backward passes:  1%| | 1/128 [00:30, value_loss=0.01]\x1b[A\r'
                    b'PPO backward passes:  1%| | 1/128 [00:30, value_loss=0.02]\r'
                    b'PPO backward passes:  2%| | 2/128 [01:00, value_loss=')
    reader = RunReader(tmp_path)
    result = reader.optimizer_progress()
    assert result['update'] == 3
    assert result['completed'] == 1
    assert result['points'][0]['value_loss'] == .02
    with log.open('ab') as f:
        f.write(b'3e-3]\r')
    result = reader.optimizer_progress()
    assert result['completed'] == 2
    assert result['points'][-1]['value_loss'] == .003
    assert len(reader.optimizer_progress()['points']) == 2
    with log.open('ab') as f:
        f.write(b'[ppo] update 4: optimizing (critic warmup)\n'
                b'PPO backward passes:  1%| | 1/128 [00:30, value_loss=0.04]\r')
    result = reader.optimizer_progress()
    assert result['update'] == 4
    assert result['completed'] == 1
    assert len(result['points']) == 1
    log.write_text('[ppo] update 5: optimizing\n')
    assert reader.optimizer_progress()['update'] == 5
    assert not reader.optimizer_progress()['points']
