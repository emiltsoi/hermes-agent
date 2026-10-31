"""Routed-platform binding: a DECLARED plugin adapter may deliver events stamped
as another platform.

Fleet carry (routed-platform-design-20261030 §4/§6): ``_binding`` resolves the
stamped platform's adapter as before, then ``_routed_deliverer`` re-binds to the
adapter that actually received the event — but only when that deliverer is
registered under its OWN platform in the same profile, appears in exactly one
registry, and names the stamped platform in ``delivers_as_platforms``. Any
deviation refuses with the same ``RuntimeStoreError('not_found')`` an undeclared
foreign stamp has always gotten, so non-declaring traffic is byte-identical.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from gateway.session_ingress_context import (
    capture_provenance, native_callback, register_transport_home,
    restore_provenance)
from hermes_state_runtime import RuntimeStoreError

MESH = Platform("mesh")


class _Adapter(BasePlatformAdapter):
    """Minimal concrete adapter: the real ``build_source`` stamps
    ``_transport_adapter_ref``; ``token`` gives the credential fingerprint a
    discoverable value like a configured adapter's."""

    def __init__(self, platform, *, token=None):
        super().__init__(PlatformConfig(enabled=True, token=token), platform)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True)

    async def get_chat_info(self, chat_id):
        return {}


def _runner(home, adapters, profile_adapters=None):
    """Bare GatewayRunner holding only what the ingress binding reads — no
    running gateway, no session DB: ``authority_for_home`` resolves the served
    home off the launch authority's db path."""
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.adapters = dict(adapters)
    runner._profile_adapters = dict(profile_adapters or {})
    register_transport_home(runner, None, home)
    runner.session_authority = SimpleNamespace(
        db=SimpleNamespace(db_path=Path(home) / "state.db"))
    return runner


def _event(source):
    return MessageEvent(
        text="ping", message_type=MessageType.TEXT, source=source,
        message_id="m1")


def _routed_source(mesh, stamped=Platform.TELEGRAM):
    return mesh.build_source(
        chat_id="7945905361", chat_type="dm", user_id="7945905361",
        user_name="emil", _platform=stamped)


def _capture(runner, event, home, profile=None):
    with native_callback(runner, event, home, profile):
        return capture_provenance(runner, event)


def _refused(runner, event, home, profile=None):
    with native_callback(runner, event, home, profile):
        with pytest.raises(RuntimeStoreError) as exc:
            capture_provenance(runner, event)
    return exc.value


def _mesh(*, declares=(Platform.TELEGRAM,)):
    adapter = _Adapter(MESH, token="mesh-cred")
    if declares is not None:
        adapter.delivers_as_platforms = declares
    return adapter


def test_declared_routed_stamp_binds_the_deliverer(tmp_path):
    """Positive: a declared deliverer's foreign stamp captures routed provenance."""
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    provenance = _capture(runner, _event(_routed_source(mesh)), tmp_path)

    assert provenance["platform"] == "telegram"
    assert provenance["delivering_platform"] == "mesh"
    assert provenance["connector"] == runner._adapter_credential_fingerprint(mesh)
    assert provenance["transport_home"] == str(Path(tmp_path).resolve())
    assert provenance["runtime_home"] == str(Path(tmp_path).resolve())


def test_undeclared_foreign_stamp_is_refused(tmp_path):
    """Requirement 3: identical setup minus ``delivers_as_platforms`` -> not_found."""
    mesh = _mesh(declares=None)
    telegram = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path)
    assert refused.reason == "not_found"


def test_declaration_without_the_stamped_platform_is_refused(tmp_path):
    """The declaration is an explicit set, not a blanket permission."""
    mesh = _mesh(declares=frozenset({Platform.DISCORD}))
    telegram = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path)
    assert refused.reason == "not_found"


def test_stamped_platform_with_no_adapter_is_refused(tmp_path):
    """The target adapter must still resolve same-profile before routed logic runs."""
    mesh = _mesh()
    runner = _runner(tmp_path, {MESH: mesh})

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path)
    assert refused.reason == "not_found"


def test_deliverer_registered_in_two_registries_is_refused(tmp_path):
    """A shared deliverer instance is ambiguous: it must appear in exactly one."""
    mesh = _mesh()
    telegram = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(
        tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh},
        profile_adapters={"beta": {MESH: mesh}})

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path)
    assert refused.reason == "not_found"


def test_deliverer_outside_the_callback_profile_is_refused(tmp_path):
    """Registered under another profile's registry: owner != profile -> not_found."""
    mesh = _mesh()
    telegram = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(
        tmp_path, {},
        profile_adapters={"alpha": {Platform.TELEGRAM: telegram},
                          "beta": {MESH: mesh}})

    refused = _refused(
        runner, _event(_routed_source(mesh)), tmp_path, profile="alpha")
    assert refused.reason == "not_found"


def test_malformed_declaration_is_refused_without_typeerror(tmp_path):
    """A string is not a collection of Platform members — refused, never iterated."""
    mesh = _mesh(declares="telegram")
    telegram = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path)
    assert refused.reason == "not_found"


def test_relay_stamp_is_refused_even_when_declared(tmp_path):
    """Relay provenance stays seal-bound; the routed path refuses before it."""
    relay = _Adapter(Platform.RELAY, token="relay-cred")
    mesh = _mesh(declares=frozenset({Platform.TELEGRAM, Platform.RELAY}))
    runner = _runner(
        tmp_path, {Platform.RELAY: relay, MESH: mesh})

    refused = _refused(runner, _event(_routed_source(mesh, stamped=Platform.RELAY)), tmp_path)
    assert refused.reason in {"not_found", "permission_denied"}


def test_restore_rebinds_the_transport_to_the_deliverer(tmp_path):
    """Round-trip: from_dict drops the weakref; the provenance hint re-binds mesh."""
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})
    source = _routed_source(mesh)
    provenance = _capture(runner, _event(source), tmp_path)

    restored = SessionSource.from_dict(source.to_dict())
    assert getattr(restored, "_transport_adapter_ref", None) is None
    runtime_home = restore_provenance(runner, restored, provenance)

    assert restored._transport_adapter_ref() is mesh
    assert str(runtime_home) == provenance["runtime_home"]


def test_tampered_delivering_platform_is_profile_mismatch(tmp_path):
    """Flipping the recorded deliverer must not re-derive the same provenance."""
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})
    source = _routed_source(mesh)
    provenance = _capture(runner, _event(source), tmp_path)

    tampered = dict(provenance)
    tampered["delivering_platform"] = "telegram"
    restored = SessionSource.from_dict(source.to_dict())
    with pytest.raises(RuntimeStoreError) as exc:
        restore_provenance(runner, restored, tampered)
    assert exc.value.reason == "profile_mismatch"


def test_garbled_delivering_platform_hint_is_refused_not_raised(tmp_path):
    """A forged hint naming no platform must refuse cleanly — never escape as ValueError.

    The hint arrives from a stored provenance dict, so it is attacker-shaped input to
    the restore path; ``Platform(hint)`` on garbage raises, and an uncaught ValueError
    here would be a 500 on intake rather than the designed ``not_found``.
    """
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})
    source = _routed_source(mesh)
    provenance = _capture(runner, _event(source), tmp_path)

    tampered = dict(provenance)
    tampered["delivering_platform"] = "not-a-platform"
    restored = SessionSource.from_dict(source.to_dict())
    with pytest.raises(RuntimeStoreError) as exc:
        restore_provenance(runner, restored, tampered)
    assert exc.value.reason == "not_found"


def test_direct_events_emit_the_unrouted_provenance_shape(tmp_path):
    """Direct unchanged: no ``delivering_platform`` key on ordinary provenance —
    identical in shape to pre-carry output for both a native telegram source and
    a native mesh source (even with the declaration present on the adapter)."""
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    direct_keys = {"transport_home", "runtime_home", "platform", "connector"}
    for adapter, platform in ((telegram, "telegram"), (mesh, "mesh")):
        source = adapter.build_source(
            chat_id="7945905361", chat_type="dm", user_id="7945905361",
            user_name="emil")
        provenance = _capture(runner, _event(source), tmp_path)
        assert set(provenance) == direct_keys
        assert provenance["platform"] == platform


def test_routed_source_derives_the_target_platform_session_key(tmp_path):
    """The whole point: a routed turn keys into the DM session, not a mesh one.

    Provenance alone is not the requirement — the requirement is which conversation the
    turn joins. Assert the derived key, and assert the unstamped contrast case really is
    mesh-keyed (otherwise this test would pass on a core that ignores the stamp).
    """
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    routed = _routed_source(mesh)
    _capture(runner, _event(routed), tmp_path)
    key = build_session_key(routed)
    assert "telegram:dm:7945905361" in key, key
    assert ":mesh:" not in key, key

    unstamped = mesh.build_source(
        chat_id="7945905361", chat_type="dm", user_id="7945905361", user_name="emil")
    mesh_key = build_session_key(unstamped)
    assert ":mesh:" in mesh_key, mesh_key
    assert mesh_key != key


def test_stamped_platform_served_by_another_profile_is_refused(tmp_path):
    """GATE-LINDA C1, in the shape it can actually occur: the stamped platform is
    served by the PROCESS (primary registry) but NOT by the profile the event is
    admitted under.

    The pre-existing same-profile resolution (``_binding`` :181-184) refuses before
    any declaration is consulted, so a routed exemption cannot widen a profile's
    authority to mint sessions for a platform it does not itself serve. The second
    half is the control: give that profile its own adapter and the same event binds
    — so the refusal is the profile scoping, not a broken fixture.
    """
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram},
                     profile_adapters={"britney": {MESH: mesh}})
    register_transport_home(runner, "britney", tmp_path)

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path, profile="britney")
    assert refused.reason == "not_found"

    brit_tg = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner._profile_adapters["britney"] = {MESH: mesh, Platform.TELEGRAM: brit_tg}
    provenance = _capture(runner, _event(_routed_source(mesh)), tmp_path, profile="britney")
    assert provenance["delivering_platform"] == "mesh"
    assert provenance["platform"] == "telegram"


def test_a_grant_is_counted_not_only_recorded(tmp_path, caplog):
    """GATE-LINDA C2: the exemption must be countable, not a grep of a provenance dict.

    One machine-parseable line per granted rebind, with a fixed token, so
    ``grep -c routed-delivery`` is the number. Direct traffic emits none.
    """
    import logging

    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    with caplog.at_level(logging.INFO, logger="gateway.session_ingress_context"):
        caplog.clear()
        _capture(runner, _event(_routed_source(mesh)), tmp_path)
        granted = [r for r in caplog.records if "routed-delivery" in r.getMessage()]
        assert len(granted) == 1, [r.getMessage() for r in granted]
        assert "platform=mesh" in granted[0].getMessage()
        assert "stamped=telegram" in granted[0].getMessage()

        caplog.clear()
        direct = telegram.build_source(
            chat_id="7945905361", chat_type="dm", user_id="7945905361", user_name="emil")
        _capture(runner, _event(direct), tmp_path)
        assert [r for r in caplog.records if "routed-delivery" in r.getMessage()] == []


def test_stamped_platform_registered_in_two_registries_is_refused(tmp_path):
    """Mutation M7: the SAME adapter object for the stamped platform in two registries.

    The pre-existing resolution refuses when the stamped platform's adapter is
    ambiguous across the primary map and a profile map. No test in this file pinned
    that for the STAMPED platform (only for the deliverer), so deleting the
    uniqueness term from :183 left the suite green — a mutation that survived.
    This test exists because that mutation was found, not because it was imagined.
    """
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh},
                     profile_adapters={"other": {Platform.TELEGRAM: telegram}})

    refused = _refused(runner, _event(_routed_source(mesh)), tmp_path)
    assert refused.reason == "not_found"

    # control: drop the duplicate registration and the identical event binds
    runner._profile_adapters["other"] = {}
    assert _capture(runner, _event(_routed_source(mesh)), tmp_path)["platform"] == "telegram"


def test_authorization_resolves_the_stamped_platform_not_the_deliverer(tmp_path):
    """Britney's finding: 0 tests referenced authorization at all.

    The security keystone is that ``_chat_scoped_grant`` asks for
    ``authorization_is_upstream`` on the STAMPED platform (:563 -> ``_adapter_flag(source.platform)``),
    so a routed turn still obeys the target platform's policy. Swap the platform for the deliverer's
    and this assertion flips: the turn would inherit mesh's upstream waiver while stamped telegram.
    """
    class _WaivingMesh(_Adapter):
        """``authorization_is_upstream`` is a read-only property on the base adapter — it must be
        overridden, not assigned, which is also how the real MeshAdapter declares it."""

        @property
        def authorization_is_upstream(self):
            return True

    mesh = _WaivingMesh(MESH, token="mesh-cred")
    mesh.delivers_as_platforms = (Platform.TELEGRAM,)
    telegram = _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    routed = _routed_source(mesh)              # stamped telegram, delivered by mesh
    assert runner._chat_scoped_grant(routed, None, False, True) is False, (
        "authorization followed the DELIVERER, not the stamp — a routed turn would inherit mesh's "
        "upstream waiver instead of telegram's policy")

    # control: the same adapter on its OWN platform still waives, so the fixture is sound
    direct = mesh.build_source(chat_id="1", chat_type="dm", user_id="1", user_name="x")
    assert runner._chat_scoped_grant(direct, None, False, True) is True


def test_the_capability_sentinel_exists_and_is_true():
    """Britney: 0 references to the sentinel anywhere. The plugin probes it at import to decide the
    stamp; rename or drop it and the fleet silently falls back to mesh-keyed sessions."""
    from gateway.session_ingress_context import ROUTED_PLATFORM_BINDING

    assert ROUTED_PLATFORM_BINDING is True, (
        "the sentinel the plugin probes is missing or false — routed delivery would silently degrade")


def test_dead_transport_ref_without_a_hint_is_refused_not_widened(tmp_path):
    """Britney's edge: a source whose transport weakref died and which carries no
    ``delivering_platform`` hint.

    ``_routed_deliverer`` sees ``delivering is None`` and returns the STAMPED platform's adapter, so
    binding alone would look like a routed event that had quietly become a direct one. The provenance
    check is what must still refuse — an event whose transport cannot be named is not an event we can
    bind, and returning a live adapter is not the same as admitting it.
    """
    mesh, telegram = _mesh(), _Adapter(Platform.TELEGRAM, token="tg-cred")
    runner = _runner(tmp_path, {Platform.TELEGRAM: telegram, MESH: mesh})

    source = _routed_source(mesh)
    source._transport_adapter_ref = lambda: None      # the deliverer is gone

    refused = _refused(runner, _event(source), tmp_path)
    assert refused.reason == "not_found", (
        f"expected capture_provenance to refuse an unnameable transport, saw {refused.reason!r}")
