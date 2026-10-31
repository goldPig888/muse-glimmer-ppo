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
