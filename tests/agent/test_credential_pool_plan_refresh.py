"""Fleet patch #22 — per-plan refresh schedule for the credential pool.

A billing-exhausted key with a known plan refresh cycle should bench until that cycle,
not re-test the same dead key every fixed TTL window (measured ~24 wasted calls/day
fleet-wide on 2026-09-15: P4 depleted, retried hourly, re-benched each time).
"""
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from agent.credential_pool import (
    _EXTRA_KEYS,
    _exhausted_ttl,
    _exhausted_until,
    _plan_refresh_spec_until,
    _plan_refresh_until,
    FAILURE_REASON_BILLING,
    FAILURE_REASON_BILLING_UNVERIFIED,
    PLAN_BENCH_MAX_SECONDS,
    PLAN_REFRESH_KEY,
    STATUS_EXHAUSTED,
    PooledCredential,
)


def entry(*, code=400, failure_reason: Optional[str] = FAILURE_REASON_BILLING, spec=None,
          status_at: Optional[float] = None):
    e = PooledCredential(
        provider="custom",
        id="p4test",
        label="P4",
        auth_type="api_key",
        priority=0,
        source="manual",
        access_token="sk-test",
    )
    e.last_status = STATUS_EXHAUSTED
    e.last_error_code = code
    e.last_status_at = status_at if status_at is not None else time.time()
    if failure_reason is not None:
        e.extra["failure_reason"] = failure_reason
    if spec is not None:
        e.extra[PLAN_REFRESH_KEY] = spec
    return e


def test_helper_sanity():
    """The fixture itself must produce a well-formed exhausted entry."""
    e = entry(spec="monthly:20")
    assert e.last_status == STATUS_EXHAUSTED
    assert e.extra[PLAN_REFRESH_KEY] == "monthly:20"


class TestSpecParsing:
    def test_absolute_epoch(self):
        now = time.time()
        got = _plan_refresh_spec_until(str(now + 3600), now=now)
        assert got is not None
        assert got == pytest.approx(now + 3600, abs=1)

    def test_absolute_iso(self):
        now = time.time()
        target = datetime.now(timezone.utc) + timedelta(hours=5)
        got = _plan_refresh_spec_until(target.isoformat(), now=now)
        assert got is not None
        assert got == pytest.approx(target.timestamp(), abs=1)

    def test_daily_is_next_midnight_utc(self):
        now = time.time()
        got = _plan_refresh_spec_until("daily", now=now)
        assert got is not None
        dt = datetime.fromtimestamp(got, timezone.utc)
        assert (dt.hour, dt.minute, dt.second) == (0, 0, 0)
        assert 0 < got - now <= 24 * 3600

    def test_daily_at_hhmm(self):
        now = time.time()
        got = _plan_refresh_spec_until("daily@08:30", now=now)
        assert got is not None
        dt = datetime.fromtimestamp(got, timezone.utc)
        assert (dt.hour, dt.minute) == (8, 30)
        assert got > now

    def test_monthly_is_next_occurrence(self):
        now = time.time()
        got = _plan_refresh_spec_until("monthly:20", now=now)
        assert got is not None
        dt = datetime.fromtimestamp(got, timezone.utc)
        assert dt.day == 20
        assert 0 < got - now <= 32 * 24 * 3600

    def test_monthly_rolls_to_next_month_when_passed(self):
        # a moment just after the 20th must resolve to the FOLLOWING month
        dt = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
        got = _plan_refresh_spec_until("monthly:20", now=dt.timestamp())
        assert got is not None
        assert datetime.fromtimestamp(got, timezone.utc).month == 10

    def test_day_clamped_to_28(self):
        now = time.time()
        got = _plan_refresh_spec_until("monthly:31", now=now)
        assert got is not None
        assert datetime.fromtimestamp(got, timezone.utc).day == 28

    @pytest.mark.parametrize("bad", ["", "   ", "fortnightly", "monthly:", "daily@99:99", None, 7])
    def test_unparseable_returns_none(self, bad):
        assert _plan_refresh_spec_until(bad, now=time.time()) is None


class TestPlanRefreshUntil:
    def test_billing_with_spec_benches_to_refresh(self):
        now = time.time()
        got = _plan_refresh_until(entry(spec="monthly:20"))
        # far beyond the 1h TTL — that is the whole point
        assert got is not None
        assert got - now > 3600

    def test_transient_throttle_ignores_spec(self):
        e = entry(code=429, failure_reason=None, spec="monthly:20")
        assert _plan_refresh_until(e) is None

    def test_billing_unverified_ignores_spec(self):
        # ambiguous billing may be a healthy credential (#82154) — keep the short cooldown
        e = entry(failure_reason=FAILURE_REASON_BILLING_UNVERIFIED, spec="monthly:20")
        assert _plan_refresh_until(e) is None

    def test_402_billing_without_failure_reason_qualifies(self):
        e = entry(code=402, failure_reason=None, spec="monthly:20")
        assert _plan_refresh_until(e) is not None

    def test_sole_credential_never_schedule_benches(self):
        # nothing to rotate to: a long bench means hard failures, not a skipped retry
        assert _plan_refresh_until(entry(spec="monthly:20"), sole_credential=True) is None

    def test_no_spec_returns_none(self):
        assert _plan_refresh_until(entry()) is None

    def test_past_absolute_returns_none(self):
        assert _plan_refresh_until(entry(spec=str(time.time() - 60))) is None

    def test_beyond_cap_falls_back(self):
        far = str(time.time() + PLAN_BENCH_MAX_SECONDS + 86400)
        assert _plan_refresh_until(entry(spec=far)) is None


class TestExhaustedUntilIntegration:
    def test_spec_overrides_the_hour_ttl(self):
        now = time.time()
        got = _exhausted_until(entry(spec="monthly:20"))
        assert got is not None
        assert got - now > 3600

    def test_no_spec_keeps_existing_ttl(self):
        e = entry()
        got = _exhausted_until(e)
        assert e.last_status_at is not None
        expected = e.last_status_at + _exhausted_ttl(
            e.last_error_code, sole_credential=False, failure_reason=FAILURE_REASON_BILLING
        )
        assert got == pytest.approx(expected, abs=1)

    def test_provider_reset_at_still_wins(self):
        # an authoritative provider-supplied reset must not be overridden by our schedule
        e = entry(spec="monthly:20")
        provider_reset = time.time() + 120
        e.last_error_reset_at = provider_reset
        assert _exhausted_until(e) == pytest.approx(provider_reset, abs=1)

    def test_non_exhausted_entry_is_untouched(self):
        e = entry(spec="monthly:20")
        e.last_status = "ok"
        assert _exhausted_until(e) is None


class TestPersistence:
    def test_key_is_registered_in_extra_whitelist(self):
        # from_dict() rehydrates extra through _EXTRA_KEYS; an unregistered key is written
        # by to_dict() and then silently DROPPED on load.
        assert PLAN_REFRESH_KEY in _EXTRA_KEYS

    def test_round_trip_survives(self):
        e = entry(spec="monthly:20")
        payload = e.to_dict()
        assert payload.get(PLAN_REFRESH_KEY) == "monthly:20"
        back = PooledCredential.from_dict("custom", payload)
        assert back.extra.get(PLAN_REFRESH_KEY) == "monthly:20"

    def test_spec_survives_a_failure_remark(self):
        # _mark_exhausted copies extra and only touches failure_reason
        e = entry(spec="monthly:20")
        e.extra["failure_reason"] = FAILURE_REASON_BILLING
        assert e.extra.get(PLAN_REFRESH_KEY) == "monthly:20"
        assert e.extra.get("failure_reason") == FAILURE_REASON_BILLING


class TestLateWindow:
    """Exact sync to the renewal, without a month-long blackout.

    A plan renews sometime DURING its day. If the bench expires at the renewal point and the
    probe fails, benching a further full cycle would write the lane off for a month. The late
    window instead retries on the short TTL until the renewal lands.
    """

    @staticmethod
    def at(monkeypatch, dt):
        import agent.credential_pool as cp
        from datetime import timezone as _tz
        monkeypatch.setattr(cp.time, "time", lambda: dt.replace(tzinfo=_tz.utc).timestamp())

    def test_mid_cycle_benches_exactly_to_the_renewal(self, monkeypatch):
        # dies 16-Sep; renewal day 20 -> bench to 20-Sep, NOT a day later
        self.at(monkeypatch, datetime(2026, 9, 16, 5, 45))
        got = _plan_refresh_until(entry(spec="monthly:20"))
        assert got is not None
        landed = datetime.fromtimestamp(got, timezone.utc)
        assert (landed.month, landed.day, landed.hour) == (9, 20, 0)

    def test_late_in_cycle_falls_back_to_the_short_ttl(self, monkeypatch):
        # the probe just failed on renewal day -> do NOT bench another month
        self.at(monkeypatch, datetime(2026, 9, 20, 0, 1))
        assert _plan_refresh_until(entry(spec="monthly:20")) is None

    def test_exact_boundary_counts_as_late(self, monkeypatch):
        self.at(monkeypatch, datetime(2026, 9, 20, 0, 0, 0))
        assert _plan_refresh_until(entry(spec="monthly:20")) is None

    def test_after_the_late_window_it_benches_the_next_cycle(self, monkeypatch):
        # >48h of failures after the renewal -> the plan did not renew
        self.at(monkeypatch, datetime(2026, 9, 23, 8, 0))
        got = _plan_refresh_until(entry(spec="monthly:20"))
        assert got is not None
        assert datetime.fromtimestamp(got, timezone.utc).month == 10

    def test_daily_late_window_is_short(self, monkeypatch):
        # daily cycle = 24h -> window is 2.4h; 3h after the reset we bench to the next one
        self.at(monkeypatch, datetime(2026, 9, 17, 3, 0))
        got = _plan_refresh_until(entry(spec="daily"))
        assert got is not None
        assert datetime.fromtimestamp(got, timezone.utc).day == 18

    def test_daily_just_after_reset_is_late(self, monkeypatch):
        self.at(monkeypatch, datetime(2026, 9, 17, 0, 1))
        assert _plan_refresh_until(entry(spec="daily")) is None

    def test_previous_helpers(self):
        from agent.credential_pool import _plan_refresh_spec_previous
        ts = datetime(2026, 9, 16, 5, 45, tzinfo=timezone.utc).timestamp()
        prev = _plan_refresh_spec_previous("monthly:20", now=ts)
        assert prev is not None
        assert datetime.fromtimestamp(prev, timezone.utc).day == 20
        assert datetime.fromtimestamp(prev, timezone.utc).month == 8
        # absolute specs have no recurrence
        assert _plan_refresh_spec_previous(str(ts - 10), now=ts) is None
