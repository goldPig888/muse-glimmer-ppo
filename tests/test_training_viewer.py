import json
from pathlib import Path
import runpy
import threading
from urllib.request import urlopen
from urllib.error import HTTPError

import pytest

module = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/serve_training_viewer.py'))
RunReader = module['RunReader']
RunCatalog = module['RunCatalog']


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


def test_catalog_keeps_distinct_runs_and_rejects_paths_outside_root(tmp_path):
    for name in ('first', 'second'):
        run = tmp_path / name
        (run / 'checkpoints/update_0001').mkdir(parents=True)
        (run / 'config.json').write_text('{"args":{"updates":2}}')
        (run / 'checkpoints/update_0001/config.json').write_text('{}')
    (tmp_path / 'empty_attempt').mkdir()
    (tmp_path / 'external').symlink_to('/tmp', target_is_directory=True)
    catalog = RunCatalog(tmp_path, training_only=False)
    assert {r['id'] for r in catalog.discover()} == {'first', 'second', 'empty_attempt'}
    assert catalog.selected('first')[0].root != catalog.selected('second')[0].root
    with pytest.raises(FileNotFoundError):
        catalog.selected('../outside')
    assert catalog.selected('empty_attempt')[0].snapshot()['kind'] == 'unknown'


def test_sft_and_server_show_their_own_metrics_not_ppo_rewards(tmp_path):
    sft = tmp_path / 'sft'
    sft.mkdir()
    (sft / 'config.json').write_text(json.dumps(dict(mode='sft', args=dict(epochs=1), train_examples=611)))
    (sft / 'progress.json').write_text('{"phase":"complete","epochs":1}')
    (sft / 'metrics.jsonl').write_text('{"epoch":0,"validation_nll":0.57}\n{"epoch":1,"training_nll":0.46,"validation_nll":0.43}\n')
    reader = RunReader(sft)
    reader.gpu = lambda: []
    state = reader.snapshot()
    assert state['kind'] == 'sft' and state['phase'] == 'complete'
    assert state['cards'][1]['value'] == .43
    assert 'mean_reward' not in state
    server = tmp_path / 'python'
    server.mkdir()
    (server / 'server_config.json').write_text('{"model":"Glimmer","load_in_4bit":true}')
    (server / 'requests.jsonl').write_text(json.dumps(dict(seconds=2, response=dict(choices=[dict(message=dict(tool_calls=[{'id':'one'}]), finish_reason='tool_calls')], usage=dict(completion_tokens=14)))) + '\n')
    reader = RunReader(server)
    reader.gpu = lambda: []
    reader.processes = type('Processes', (), {'match': lambda self, root: 123})()
    state = reader.snapshot()
    assert state['kind'] == 'python_server' and state['running'] and state['pid'] == 123
    assert state['cards'][1]['value'] == 1
    assert state['series'][0]['rows'][0]['completion_tokens'] == 14
    assert not state['metrics']


def test_default_catalog_includes_training_only(tmp_path):
    configs = {
        'ppo_run': ('config.json', {'args': {'updates': 2}}),
        'sft_run': ('config.json', {'mode': 'sft', 'args': {'epochs': 1}}),
        'server': ('server_config.json', {'model': 'Glimmer'}),
        'evaluation': ('evaluation_config.json', {}),
        'physics_check': ('results.json', [{'success': True}]),
    }
    for name, (filename, config) in configs.items():
        folder = tmp_path / name
        folder.mkdir()
        (folder / filename).write_text(json.dumps(config))
    assert {r['id'] for r in RunCatalog(tmp_path).discover()} == {'ppo_run', 'sft_run'}
    assert len(RunCatalog(tmp_path, training_only=False).discover()) == len(configs)


def test_evaluation_and_checkpoint_comparison_preserve_failures(tmp_path):
    (tmp_path / 'evaluation_config.json').write_text('{}')
    (tmp_path / 'summary.json').write_text('{"episodes":4,"success_rate":0}')
    (tmp_path / 'results.jsonl').write_text(''.join(json.dumps(dict(init=i, success=False, outcome='step_limit'))+'\n' for i in range(4)))
    reader = RunReader(tmp_path)
    reader.gpu = lambda: []
    state = reader.snapshot()
    assert state['phase'] == 'complete' and state['record_count'] == 4
    assert state['cards'][2]['value'] == '0.0%'
    assert state['summary']['success_rate'] == 0
    (tmp_path / 'evaluation_config.json').unlink()
    (tmp_path / 'config.json').write_text('{"checkpoints":["update_0011","update_0012"]}')
    (tmp_path / 'summary.json').write_text('{"checkpoints":[{"success_rate":0.8},{"success_rate":0.8}]}')
    state = reader.snapshot()
    assert state['kind'] == 'comparison'
    assert len(state['summary']['checkpoints']) == 2


def test_multi_run_http_state_and_frames_cannot_leak_across_runs(tmp_path):
    for name in ('first', 'second'):
        run = tmp_path / name
        frame = run / 'episodes/episode_0001/frames/call01_agentview.png'
        frame.parent.mkdir(parents=True)
        frame.write_bytes(name.encode())
        (run / 'config.json').write_text('{"args":{"updates":1}}')
    catalog = RunCatalog(tmp_path)
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(('127.0.0.1', 0), module['make_handler'](catalog))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        for name in ('first', 'second', 'first'):
            with urlopen(endpoint + '/api/state?run=' + name) as response:
                state = json.load(response)
            assert state['run'] == name and state['run_id'] == name
            frame = state['frames']['agentview']['path']
            with urlopen(endpoint + '/frames/' + frame + '?run=' + name) as response:
                assert response.read() == name.encode()
        for path in ('/api/state?run=../outside', '/frames/../second/config.json?run=first'):
            with pytest.raises(HTTPError) as error:
                urlopen(endpoint + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
