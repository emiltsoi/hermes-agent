"""CommandCode's reasoning-effort vocabulary, and the custom-endpoint path that must declare it.

The fleet reaches CommandCode through a *custom* provider entry (``goat-fleet`` ->
``https://api.commandcode.ai/provider/v1``), so the request is built by the generic
``custom`` profile. That profile clamped to ``OPENAI_COMPAT_WIRE_EFFORTS``, which is WIDER
than CommandCode's: ``none`` and ``minimal`` were emitted verbatim and the endpoint answered

    Error code: 400 - Invalid option: expected one of "low"|"medium"|"high"|"xhigh"|"max"
    (param=reasoning_effort)

A 400 on the reasoning field is not cosmetic — the turn falls through to the next lane, so a
level CommandCode rejects silently moved the request off the credential pool onto
deepseek-direct. Per the module doctrine ("when a provider rejects a level, fix its declared
set, never a predicate"), the vocabulary now travels with the endpoint.
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent.reasoning_effort import (  # noqa: E402
    COMMANDCODE_EFFORTS,
    OPENAI_COMPAT_WIRE_EFFORTS,
    clamp_effort,
)


def _load_custom_profile():
    """The provider plugin lives under a hyphenated dir, so it is not importable by name."""
    spec = importlib.util.spec_from_file_location(
        "hermes_test_custom_provider", REPO / "plugins" / "model-providers" / "custom" / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


custom_mod = _load_custom_profile()
PROFILE = custom_mod.custom

COMMANDCODE_URL = "https://api.commandcode.ai/provider/v1"
OLLAMA_URL = "http://192.168.1.48:11434"
GENERIC_URL = "https://api.deepseek.com/v1"

#: What the endpoint itself publishes in its 400 message.
WIRE_ACCEPTS = {"low", "medium", "high", "xhigh", "max"}


def _emitted(profile, *, effort=None, enabled=None, base_url):
    cfg = {}
    if effort is not None:
        cfg["effort"] = effort
    if enabled is not None:
        cfg["enabled"] = enabled
    _, top_level = profile.build_api_kwargs_extras(reasoning_config=cfg or None, base_url=base_url)
    return top_level.get("reasoning_effort")


class TestVocabulary:
    def test_commandcode_set_is_narrower_than_openai_compat(self):
        assert set(COMMANDCODE_EFFORTS) < set(OPENAI_COMPAT_WIRE_EFFORTS)

    def test_commandcode_set_excludes_the_two_it_400s_on(self):
        assert "none" not in COMMANDCODE_EFFORTS
        assert "minimal" not in COMMANDCODE_EFFORTS
        assert set(COMMANDCODE_EFFORTS) == WIRE_ACCEPTS

    def test_none_clamps_to_lowest_supported(self):
        # "none" cannot be expressed on this wire; the weakest level is the closest honest match.
        assert clamp_effort("none", COMMANDCODE_EFFORTS) == "low"

    def test_minimal_clamps_to_lowest_supported(self):
        assert clamp_effort("minimal", COMMANDCODE_EFFORTS) == "low"


class TestDetector:
    def test_matches_apex_and_subdomains(self):
        assert custom_mod._looks_like_commandcode_endpoint(COMMANDCODE_URL)
        assert custom_mod._looks_like_commandcode_endpoint("https://api.commandcode.ai/v1")
        assert custom_mod._looks_like_commandcode_endpoint("api.commandcode.ai")

    def test_rejects_other_hosts_and_empty(self):
        for url in (None, "", "   ", OLLAMA_URL, GENERIC_URL, "https://notcommandcode.ai/v1"):
            assert not custom_mod._looks_like_commandcode_endpoint(url)


class TestWireLevelsOnCommandCode:
    """Every level Hermes can hold must land inside what the endpoint accepts."""

    def test_every_ladder_level_is_emitted_within_the_accepted_set(self):
        for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
            emitted = _emitted(PROFILE, effort=level, base_url=COMMANDCODE_URL)
            assert emitted in WIRE_ACCEPTS, f"{level} emitted {emitted!r}, which 400s on this wire"

    def test_disabled_reasoning_does_not_emit_none(self):
        # The old code hardcoded "none" here, which is precisely the 400 shape.
        assert _emitted(PROFILE, enabled=False, base_url=COMMANDCODE_URL) in WIRE_ACCEPTS

    def test_supported_levels_pass_through_verbatim(self):
        for level in ("low", "medium", "high", "xhigh", "max"):
            assert _emitted(PROFILE, effort=level, base_url=COMMANDCODE_URL) == level


class TestNegativeControls:
    """A fix that narrows every custom endpoint would be a different bug."""

    def test_ollama_still_disables_with_none_and_the_think_flag(self):
        extra_body, top_level = PROFILE.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}, base_url=OLLAMA_URL
        )
        assert top_level.get("reasoning_effort") == "none"
        assert extra_body.get("think") is False

    def test_generic_custom_endpoint_keeps_the_wide_vocabulary(self):
        assert _emitted(PROFILE, effort="none", base_url=GENERIC_URL) == "none"
        assert _emitted(PROFILE, effort="minimal", base_url=GENERIC_URL) == "minimal"

    def test_no_base_url_falls_back_to_the_generic_set(self):
        # Unknown endpoint -> wide set. Never guess a narrow vocabulary from a missing URL.
        assert _emitted(PROFILE, effort="none", base_url=None) == "none"
