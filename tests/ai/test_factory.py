from __future__ import annotations

import pytest

from chibi.ai.providers.anthropic_provider import AnthropicProvider
from chibi.ai.providers.factory import (
    DEFAULT_MODELS,
    build_provider,
    env_key_available,
    is_cloud,
    needs_key,
    stored_key,
)
from chibi.ai.providers.fake import FakeProvider
from chibi.ai.providers.ollama_provider import OllamaProvider
from chibi.ai.providers.openai_provider import OpenAIProvider
from chibi.ai.secrets import MemorySecretStore, SecretStoreError
from chibi.ai.types import (
    CancelledError,
    CancelToken,
    ChatTurn,
    GenerationRequest,
    ProviderRequestError,
    ProviderUnavailableError,
    Purpose,
    Role,
    TextProvider,
)
from chibi.core.config import AISettings


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_none_builds_nothing():
    assert build_provider(AISettings(provider="none"), MemorySecretStore()) is None


def test_anthropic_uses_stored_key_default_model_and_settings():
    secrets = MemorySecretStore({"anthropic_api_key": "sk-ant-stored"})
    p = build_provider(AISettings(provider="anthropic", effort="high", request_timeout_seconds=45), secrets)
    assert isinstance(p, AnthropicProvider)
    assert p.model == DEFAULT_MODELS["anthropic"] == "claude-opus-5-5"
    assert p.effort == "high"
    assert p._client.api_key == "sk-ant-stored"
    assert p._client.timeout == 45
    assert p._client.max_retries == 0  # the service owns retries
    p2 = build_provider(AISettings(provider="anthropic", model="claude-haiku-5-5"), secrets)
    assert p2.model == "claude-haiku-5-5"


def test_openai_requires_a_model():
    with pytest.raises(ProviderRequestError) as err:
        build_provider(AISettings(provider="openai"), MemorySecretStore())
    assert "Вибери модель" in str(err.value)


def test_openai_and_ollama():
    secrets = MemorySecretStore({"openai_api_key": "sk-o"})
    p = build_provider(AISettings(provider="openai", model="gpt-5-mini",
                                  openai_base_url="http://localhost:1234/v1"), secrets)
    assert isinstance(p, OpenAIProvider) and p.model == "gpt-5-mini" and p.base_url == "http://localhost:1234/v1"
    assert p.has_key
    o = build_provider(AISettings(provider="ollama"), secrets)
    assert isinstance(o, OllamaProvider) and o.model == "" and o.base_url == "http://localhost:11434"
    remote = AISettings(provider="ollama", model="gemma3:4b", ollama_url="http://10.0.0.5:11434")
    o2 = build_provider(remote, secrets)
    assert o2.model == "gemma3:4b" and o2.is_local is False


@pytest.mark.parametrize("kwargs,cloud", [
    ({"provider": "none"}, False),
    ({"provider": "anthropic"}, True),
    ({"provider": "openai"}, True),
    ({"provider": "openai", "openai_base_url": "http://127.0.0.1:1234/v1"}, False),
    ({"provider": "openai", "openai_base_url": "http://192.168.1.2:1234/v1"}, True),
    ({"provider": "ollama"}, False),
    ({"provider": "ollama", "ollama_url": "http://[::1]:11434"}, False),
    ({"provider": "ollama", "ollama_url": "https://my-gpu-box.example.com"}, True),
])
def test_is_cloud(kwargs, cloud):
    assert is_cloud(AISettings(**kwargs)) is cloud


@pytest.mark.parametrize("kwargs,required", [
    ({"provider": "anthropic"}, True),
    ({"provider": "openai"}, True),
    ({"provider": "openai", "openai_base_url": "http://localhost:1234/v1"}, False),
    ({"provider": "openai", "openai_base_url": "http://192.168.1.2:1234/v1"}, False),
    ({"provider": "ollama"}, False),
    ({"provider": "none"}, False),
])
def test_needs_key(kwargs, required):
    assert needs_key(AISettings(**kwargs)) is required


def test_env_key_available(monkeypatch):
    assert not env_key_available(AISettings(provider="anthropic"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
    assert env_key_available(AISettings(provider="anthropic"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o-env")
    assert env_key_available(AISettings(provider="openai"))
    assert not env_key_available(AISettings(provider="openai", openai_base_url="https://api.openai.com.evil.io/v1"))
    assert not env_key_available(AISettings(provider="ollama"))


def test_stored_key_survives_a_broken_store(caplog):
    class Broken(MemorySecretStore):
        def get(self, name):
            raise SecretStoreError("CredReadW failed (error 5)")

    assert stored_key(Broken(), "anthropic") is None
    assert stored_key(MemorySecretStore({"openai_api_key": " k "}), "openai") == "k"
    assert stored_key(MemorySecretStore(), "ollama") is None


# -- fake provider ---------------------------------------------------------------------------------


def _req() -> GenerationRequest:
    return GenerationRequest(system="s", turns=(ChatTurn(Role.USER, "привіт"),), purpose=Purpose.CHAT)


def test_fake_provider_scripted_replies():
    fake = FakeProvider(["Привіт, як справи?", ProviderUnavailableError("down")], default_reply=None)
    assert isinstance(fake, TextProvider) and fake.is_local
    pieces: list[str] = []
    result = fake.stream(_req(), pieces.append)
    assert "".join(pieces) == result.text == "Привіт, як справи?"
    assert len(pieces) == 3
    with pytest.raises(ProviderUnavailableError):
        fake.generate(_req())
    with pytest.raises(ProviderUnavailableError):
        fake.generate(_req())  # script exhausted
    assert len(fake.requests) == 3


def test_fake_provider_gate_honours_cancel():
    import threading

    gate = threading.Event()
    fake = FakeProvider(["x"], gate=gate)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        fake.generate(_req(), token)
    gate.set()
    assert fake.generate(_req()).text == "Привіт! Я тут."
