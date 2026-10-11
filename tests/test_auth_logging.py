"""Credential values must not be emitted during auth startup or reload failures."""

import asyncio
import json
import traceback
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from common import auth


API_KEYS = ["synthetic-api-secret-a", "synthetic-api-secret-b"]
ADMIN_KEY = "synthetic-admin-secret"
ALL_SECRETS = API_KEYS + [ADMIN_KEY, "nested-synthetic-secret"]


def assert_no_secrets(value):
    text = str(value)
    for secret in ALL_SECRETS:
        assert secret not in text


@pytest.fixture
def auth_state(monkeypatch):
    messages = []
    capture = SimpleNamespace(info=messages.append, warning=messages.append)
    monkeypatch.setattr(auth, "logger", capture)
    monkeypatch.setattr(auth, "xlogger", capture)
    monkeypatch.setattr(auth, "AUTH_KEYS", None)
    monkeypatch.setattr(auth, "DISABLE_AUTH", False)
    # Avoid a background watcher in startup tests; watcher behavior is tested below.
    monkeypatch.setattr(auth, "_watch_task", object())
    monkeypatch.setattr(auth, "_reload_lock", asyncio.Lock())
    return messages


@pytest.mark.parametrize("api_key", [API_KEYS[0], API_KEYS, [API_KEYS[0], API_KEYS[0]]])
def test_startup_logs_path_and_count_without_credentials(monkeypatch, auth_state, api_key):
    keys = auth.AuthKeys(api_key=api_key, admin_key=ADMIN_KEY)
    monkeypatch.setattr(auth, "_read_auth_file", AsyncMock(return_value=keys))
    asyncio.run(auth.load_auth_keys(False))
    assert auth.AUTH_KEYS is keys
    assert auth.DISABLE_AUTH is False
    assert auth_state == [
        f"Authentication enabled: api_tokens.yml ({len(keys._api_key_set)} API key(s), 1 admin key)."
    ]
    assert_no_secrets(auth_state)
    assert keys.verify_key(ADMIN_KEY, "admin_key")
    assert keys.verify_key(ADMIN_KEY, "api_key")
    assert keys.verify_key(API_KEYS[0], "api_key")
    assert not keys.verify_key(API_KEYS[0], "admin_key")


def test_generated_keys_are_persisted_but_not_logged(tmp_path, monkeypatch, auth_state):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(auth.secrets, "token_hex", lambda _: next(values))
    values = iter([API_KEYS[0], ADMIN_KEY])
    asyncio.run(auth.load_auth_keys(False))
    saved = asyncio.run(auth._read_auth_file())
    assert saved.api_key == API_KEYS[0]
    assert saved.admin_key == ADMIN_KEY
    assert_no_secrets(auth_state)
    assert len(auth_state) == 1


def test_credentials_are_hidden_inside_nested_model_representations():
    keys = auth.AuthKeys(api_key=API_KEYS, admin_key=ADMIN_KEY)
    nested = {"credentials": [keys, {"repeated": keys}]}
    assert_no_secrets(repr(keys))
    assert_no_secrets(str(keys))
    assert_no_secrets(repr(nested))
    # Explicit serialization is required for the existing credential file format.
    assert keys.model_dump() == {"api_key": API_KEYS, "admin_key": ADMIN_KEY}


def test_nested_invalid_input_is_hidden_in_validation_message():
    invalid = {"api_key": [{"token": ALL_SECRETS[-1]}], "admin_key": {"value": ADMIN_KEY}}
    with pytest.raises(ValidationError) as error:
        auth.AuthKeys.model_validate(invalid)
    assert_no_secrets(str(error.value))
    assert "api_key" in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": [{"token": ALL_SECRETS[-1]}], "admin_key": {"value": ADMIN_KEY}},
        {"api_key": API_KEYS},
        {"api_key": API_KEYS, "admin_key": [ADMIN_KEY]},
    ],
)
def test_invalid_auth_file_raises_a_sanitized_error(tmp_path, payload):
    path = tmp_path / "api_tokens.yml"
    path.write_text(json.dumps(payload))
    with pytest.raises(auth.AuthFileError) as error:
        asyncio.run(auth._read_auth_file(str(path)))
    assert "api_tokens.yml (ValidationError)" in str(error.value)
    assert_no_secrets(str(error.value))
    rendered = "".join(traceback.format_exception(error.type, error.value, error.tb))
    assert_no_secrets(rendered)


def test_yaml_parse_failure_does_not_disclose_source_lines(tmp_path):
    path = tmp_path / "api_tokens.yml"
    path.write_text(f"api_key: [{API_KEYS[0]}\nadmin_key: {ADMIN_KEY}\n")
    with pytest.raises(auth.AuthFileError) as error:
        asyncio.run(auth._read_auth_file(str(path)))
    assert "api_tokens.yml" in str(error.value)
    assert_no_secrets(str(error.value))
    assert_no_secrets("".join(traceback.format_exception(error.type, error.value, error.tb)))


def test_missing_file_still_uses_generation_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(auth._read_auth_file(str(tmp_path / "api_tokens.yml")))


@pytest.mark.parametrize("succeeds", [False, True])
def test_reload_preserves_auth_state_and_never_logs_error_payload(
    monkeypatch, auth_state, succeeds
):
    old = auth.AuthKeys(api_key=API_KEYS[0], admin_key=ADMIN_KEY)
    new = auth.AuthKeys(api_key=API_KEYS, admin_key=ADMIN_KEY)
    monkeypatch.setattr(auth, "AUTH_KEYS", old)
    if succeeds:
        reader = AsyncMock(return_value=new)
    else:
        # Simulate an exception from a library that embeds a complete nested token object.
        reader = AsyncMock(
            side_effect=ValueError(
                repr({"api_key": [{"token": ALL_SECRETS[-1]}], "admin_key": ADMIN_KEY})
            )
        )
    monkeypatch.setattr(auth, "_read_auth_file", reader)
    sleeps = 0

    async def one_iteration(_):
        nonlocal sleeps
        sleeps += 1
        if sleeps > 1:
            raise asyncio.CancelledError

    with (
        patch.object(
            auth.os, "stat", side_effect=[SimpleNamespace(st_mtime=1), SimpleNamespace(st_mtime=2)]
        ),
        patch.object(auth.asyncio, "sleep", one_iteration),
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(auth._watch_auth_file())
    reader.assert_awaited_once()
    assert auth.AUTH_KEYS is (new if succeeds else old)
    assert len(auth_state) == 1
    assert_no_secrets(auth_state)
    if succeeds:
        assert auth_state == ["Reloaded auth keys from api_tokens.yml (2 API key(s))."]
    else:
        assert auth_state == [
            "Failed to reload api_tokens.yml, keeping the previous keys (ValueError)."
        ]
