"""Custom / Ollama (local) provider profile: any endpoint registered as
provider="custom" (Ollama, vLLM, llama.cpp, GLM-5.2 on ARK, …)."""

from typing import Any
from urllib.parse import urlparse

from agent.reasoning_effort import COMMANDCODE_EFFORTS, OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


def _looks_like_ollama_endpoint(base_url: str | None) -> bool:
    """True only for explicit Ollama signatures (port 11434 or an ``ollama`` host label).
    ``think`` is Ollama-native; strict hosts (Mistral, Groq) 422 on it, and
    arbitrary localhost may be llama.cpp / vLLM / LM Studio."""
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    try:  # urlparse raises ValueError on malformed ports ("host:99999"); treat as not-Ollama.
        if parsed.port == 11434:
            return True
    except ValueError:
        return False
    host = (parsed.hostname or "").lower().rstrip(".")
    return bool(host) and (host == "ollama.com" or host.endswith(".ollama.com") or "ollama" in host.split("."))


def _looks_like_commandcode_endpoint(base_url: str | None) -> bool:
    """True for CommandCode's OpenAI-compatible endpoint (``api.commandcode.ai``).

    CommandCode accepts exactly ``low|medium|high|xhigh|max`` and 400s on ``none``/``minimal``.
    That 400 is not cosmetic: the turn falls through to the next lane, so a level this endpoint
    rejects silently *relocates* the request to another provider. Its declared set therefore has
    to travel with the endpoint rather than with the generic OpenAI-compat default — same shape
    as the Ollama check above, for the same reason (the wire's vocabulary is a property of the
    host, not of "custom").
    """
    raw = (base_url or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"//{raw}")
    host = (parsed.hostname or "").lower().rstrip(".")
    return bool(host) and (host == "api.commandcode.ai" or host.endswith(".commandcode.ai"))


class CustomProfile(ProviderProfile):
    """Custom/Ollama local provider — think=false and num_ctx support."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, ollama_num_ctx: int | None = None, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        if ollama_num_ctx:
            extra_body["options"] = {"num_ctx": ollama_num_ctx}
        # disabled -> top-level reasoning_effort="none" (Ollama's /v1 ignores
        # extra_body.think) plus think=False only on Ollama URLs; enabled+effort ->
        # top-level reasoning_effort clamped to the OpenAI-compat wire (GLM/ARK,
        # vLLM and SGLang all top out at "max"; "ultra" verbatim 400s); enabled
        # without effort -> omit so the server default applies. Never emit
        # think=True (Ollama-only flag).
        if reasoning_config and isinstance(reasoning_config, dict):
            base_url = ctx.get("base_url")
            # The wire's vocabulary is a property of the endpoint, not of "custom".
            supported = (
                COMMANDCODE_EFFORTS if _looks_like_commandcode_endpoint(base_url)
                else OPENAI_COMPAT_WIRE_EFFORTS
            )
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if effort == "none" or reasoning_config.get("enabled", True) is False:
                # See #14820. "none" stays "none" where the wire publishes it as a level
                # (Ollama disables that way). On a declared set WITHOUT it, clamp_effort lands on
                # the weakest supported level — the closest honest expression of "as little
                # thinking as this endpoint allows", and strictly better than a 400 that silently
                # relocates the turn to another provider.
                top_level["reasoning_effort"] = clamp_effort("none", supported)
                if _looks_like_ollama_endpoint(base_url):
                    extra_body["think"] = False
            elif effort:
                top_level["reasoning_effort"] = clamp_effort(effort, supported)
        return extra_body, top_level

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """base_url is user-configured; fetch only if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom", aliases=("ollama", "local", "vllm", "llamacpp", "llama.cpp", "llama-cpp"),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # An arbitrary client ceiling can exceed a local server's actual output limit.
    # The endpoint owns its generation default.
)

register_provider(custom)
