import pytest

from glimmer_ppo.rollout_server import MEDIA_MARKER, parse_completion, to_server_prompt


def test_image_placeholders_become_media_markers():
    text = 'goal <|image|> agent <|image|> wrist'
    assert to_server_prompt(text, '<|image|>', 2) == f'goal {MEDIA_MARKER} agent {MEDIA_MARKER} wrist'


def test_placeholder_count_must_match_images():
    with pytest.raises(ValueError):
        to_server_prompt('only <|image|> one', '<|image|>', 2)


def test_parse_completion_new_and_old_llama_cpp():
    ids, text, truncated = parse_completion({'tokens': [5, 6, 7], 'content': 'abc', 'stop_type': 'eos'}, 100)
    assert ids == [5, 6, 7] and text == 'abc' and not truncated
    _, _, truncated = parse_completion({'tokens': [1] * 100, 'content': 'x', 'stop_type': 'limit'}, 100)
    assert truncated
    _, _, truncated = parse_completion({'tokens': [1, 2], 'content': 'x', 'stopped_limit': True}, 100)
    assert truncated


def test_missing_token_ids_is_an_error():
    with pytest.raises(RuntimeError):
        parse_completion({'content': 'abc'}, 10)
