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
