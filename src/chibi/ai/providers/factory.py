"""Builds the configured :class:`TextProvider` from :class:`AISettings`.

Also decides whether a configuration is *cloud* (data leaves the PC and needs
``Permission.CLOUD_AI``) and whether it needs an API key.  Anthropic is always
cloud; OpenAI-compatible servers and Ollama are cloud unless their host is
this machine.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from chibi.ai.providers.anthropic_provider import AnthropicProvider
from chibi.ai.providers.ollama_provider import OllamaProvider
from chibi.ai.providers.openai_provider import (
    OPENAI_HOST,
    OpenAIProvider,
    is_loopback_url,
    is_private_network_url,
)
from chibi.ai.secrets import KEY_NAMES, SecretStore
from chibi.ai.types import ProviderRequestError, TextProvider
from chibi.core.config import AISettings

log = logging.getLogger(__name__)

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}
#: Environment variables the SDKs / servers accept instead of a stored key.
KEY_ENV_VARS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
}


def is_cloud(ai: AISettings) -> bool:
    """True when requests leave this PC (``Permission.CLOUD_AI`` required)."""
    if ai.provider == "anthropic":
        return True
    if ai.provider == "openai":
        return not is_loopback_url(ai.openai_base_url)
    if ai.provider == "ollama":
        return not is_loopback_url(ai.ollama_url)
    return False


def needs_key(ai: AISettings) -> bool:
    """Whether the provider can't work without an API key.

    Local / LAN OpenAI-compatible servers (LM Studio, llama.cpp) usually run
    without one, so only a public host requires it.
    """
    if ai.provider == "anthropic":
        return True
    if ai.provider == "openai":
        return not is_private_network_url(ai.openai_base_url)
    return False


def env_key_available(ai: AISettings) -> bool:
    """A key in the environment the provider would use (OpenAI's only for api.openai.com)."""
    names = KEY_ENV_VARS.get(ai.provider, ())
    if ai.provider == "openai" and (urlsplit(ai.openai_base_url).hostname or "").lower() != OPENAI_HOST:
        return False
    return any((os.environ.get(name) or "").strip() for name in names)


def stored_key(secrets: SecretStore, provider: str) -> str | None:
    """The stored key for *provider*, or None (also when the store fails - logged, never the value)."""
    name = KEY_NAMES.get(provider)
    if name is None:
        return None
    try:
        value = secrets.get(name)
    except Exception as exc:  # an OS credential store can fail in many ways
        log.warning("reading the %s key failed (%s)", provider, type(exc).__name__)
        return None
    return value.strip() if value and value.strip() else None


def build_provider(ai: AISettings, secrets: SecretStore) -> TextProvider | None:
    """The provider for *ai*, or None when AI is off.

    Raises :class:`ProviderRequestError` (Ukrainian ``user_message``) for a
    configuration that can't work, e.g. an OpenAI-compatible server without a
    model name.
    """
    if ai.provider == "anthropic":
        # The service is the single retry layer (interruptible backoff, honours
        # cancellation); SDK-internal retries would multiply the worst-case wait.
        return AnthropicProvider(stored_key(secrets, "anthropic"), ai.model or DEFAULT_MODELS["anthropic"],
                                 effort=ai.effort, timeout=ai.request_timeout_seconds, max_retries=0)
    if ai.provider == "openai":
        if not ai.model:
            raise ProviderRequestError("Вибери модель",
                                       user_message="Вибери модель для OpenAI-сумісного сервера.")
        return OpenAIProvider(stored_key(secrets, "openai"), ai.openai_base_url, ai.model,
                              timeout=ai.request_timeout_seconds)
    if ai.provider == "ollama":
        # Local models can take a while to load on first use: at least 60 s.
        return OllamaProvider(ai.ollama_url, ai.model or "", timeout=max(60.0, ai.request_timeout_seconds))
    return None
