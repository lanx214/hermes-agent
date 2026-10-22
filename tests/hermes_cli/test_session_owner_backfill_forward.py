"""A standalone dashboard forwards owner-backfill to a live Gateway (#106742)."""
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from fastapi import HTTPException, Request


def _endpoint():
    return SimpleNamespace(
        api_origin="http://127.0.0.1:8642",
        profile_id="/tmp/hermes/profiles/work",
        instance_id="owner-instance",
    )


@pytest.mark.asyncio
async def test_live_owner_receives_native_http_backfill_forward(tmp_path, monkeypatch):
    from hermes_cli import web_server_sessions

    home = tmp_path / "profiles" / "work"
    endpoint = _endpoint()
    calls = {}
    monkeypatch.setattr("hermes_cli.web_server_cron._cron_profile_home", lambda profile: ("work", home))
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime.discover_gateway_endpoint",
        lambda discovered_home, timeout: SimpleNamespace(state="ready", endpoint=endpoint),
    )

    def mint_ticket(ticket_home, ticket_endpoint, *, purpose):
        calls["ticket"] = (ticket_home, ticket_endpoint, purpose)
        return "ephemeral-test-ticket"

    monkeypatch.setattr("hermes_cli.gateway_client._session_ticket", mint_ticket)

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"ok": True, "stamped": 0, "profile": "work"}

    class Client:
        def __init__(self, *, timeout, trust_env):
            calls["client_options"] = (timeout, trust_env)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json, headers):
            calls["post"] = (url, json, headers)
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)

    result = await web_server_sessions._backfill_via_gateway_owner("work", "work")

    assert result == {"ok": True, "stamped": 0, "profile": "work"}
    assert calls["ticket"] == (home, endpoint, "native-http")
    assert calls["client_options"] == (10.0, False)
    assert calls["post"] == (
        "http://127.0.0.1:8642/api/sessions/owner-backfill",
        {"profile": "work"},
        {"X-Hermes-Gateway-Ticket": "ephemeral-test-ticket"},
    )


@pytest.mark.asyncio
async def test_absent_owner_allows_offline_fallback(tmp_path, monkeypatch):
    from hermes_cli import web_server_sessions

    home = tmp_path / "home"
    monkeypatch.setattr("hermes_cli.web_server_cron._cron_profile_home", lambda profile: ("default", home))
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime.discover_gateway_endpoint",
        lambda discovered_home, timeout: SimpleNamespace(state="absent", endpoint=None),
    )

    assert await web_server_sessions._backfill_via_gateway_owner(None, "default") is None


@pytest.mark.asyncio
async def test_live_but_unreachable_owner_is_retryable_not_offline_fallback(tmp_path, monkeypatch):
    from hermes_cli import web_server_sessions

    home = tmp_path / "home"
    monkeypatch.setattr("hermes_cli.web_server_cron._cron_profile_home", lambda profile: ("default", home))
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime.discover_gateway_endpoint",
        lambda discovered_home, timeout: SimpleNamespace(state="starting", endpoint=None),
    )

    with pytest.raises(HTTPException) as exc:
        await web_server_sessions._backfill_via_gateway_owner(None, "default")

    assert exc.value.status_code == 503
    assert exc.value.detail == "owner_unreachable"


@pytest.mark.asyncio
async def test_standalone_route_forwards_before_offline_write(monkeypatch):
    from hermes_cli.web_models import SessionOwnerBackfill
    from hermes_cli.web_routers import sessions

    expected = {"ok": True, "stamped": 0, "profile": "work"}
    monkeypatch.setattr(sessions, "_serving_profile", lambda profile: "work")
    monkeypatch.setattr(sessions, "_owner_db_for", lambda request, profile: None)

    async def forward(profile, stamp):
        assert (profile, stamp) == ("work", "work")
        return expected

    monkeypatch.setattr(sessions, "_backfill_via_gateway_owner", forward)
    monkeypatch.setattr(sessions, "_with_db", lambda *args, **kwargs: pytest.fail("offline writer must not run"))

    result = await sessions.backfill_session_owner_profiles(SessionOwnerBackfill(profile="work"), cast(Request, SimpleNamespace()))

    assert result == expected


@pytest.mark.asyncio
async def test_standalone_route_keeps_offline_backfill_when_owner_is_absent(monkeypatch):
    from hermes_cli.web_models import SessionOwnerBackfill
    from hermes_cli.web_routers import sessions

    calls = []
    monkeypatch.setattr(sessions, "_serving_profile", lambda profile: "default")
    monkeypatch.setattr(sessions, "_owner_db_for", lambda request, profile: None)

    async def no_owner(profile, stamp):
        return None

    def offline_write(profile, operation, *, read_only):
        calls.append((profile, read_only))
        db = SimpleNamespace(backfill_null_session_profiles=lambda stamp: 2)
        return operation(db)

    monkeypatch.setattr(sessions, "_backfill_via_gateway_owner", no_owner)
    monkeypatch.setattr(sessions, "_with_db", offline_write)

    result = await sessions.backfill_session_owner_profiles(
        SessionOwnerBackfill(), cast(Request, SimpleNamespace()))

    assert result == {"ok": True, "stamped": 2, "profile": "default"}
    assert calls == [(None, False)]


@pytest.mark.asyncio
async def test_lock_race_retries_owner_discovery(monkeypatch):
    from hermes_cli.web_models import SessionOwnerBackfill
    from hermes_cli.web_routers import sessions

    expected = {"ok": True, "stamped": 0, "profile": "work"}
    attempts = 0
    monkeypatch.setattr(sessions, "_serving_profile", lambda profile: "work")
    monkeypatch.setattr(sessions, "_owner_db_for", lambda request, profile: None)

    async def forward(profile, stamp):
        nonlocal attempts
        attempts += 1
        return None if attempts == 1 else expected

    def offline_conflict(*args, **kwargs):
        raise HTTPException(
            status_code=409,
            detail="Exclusive maintenance refused: Gateway runtime already owns profile /root/.hermes",
        )

    monkeypatch.setattr(sessions, "_backfill_via_gateway_owner", forward)
    monkeypatch.setattr(sessions, "_with_db", offline_conflict)

    result = await sessions.backfill_session_owner_profiles(SessionOwnerBackfill(profile="work"), cast(Request, SimpleNamespace()))

    assert result == expected
    assert attempts == 2


@pytest.mark.asyncio
async def test_unrelated_409_is_not_retried_as_owner_conflict(monkeypatch):
    from hermes_cli.web_models import SessionOwnerBackfill
    from hermes_cli.web_routers import sessions

    attempts = 0
    monkeypatch.setattr(sessions, "_serving_profile", lambda profile: "work")
    monkeypatch.setattr(sessions, "_owner_db_for", lambda request, profile: None)

    async def absent_owner(profile, stamp):
        nonlocal attempts
        attempts += 1
        return None

    def unrelated_conflict(*args, **kwargs):
        raise HTTPException(status_code=409, detail="session_busy")

    monkeypatch.setattr(sessions, "_backfill_via_gateway_owner", absent_owner)
    monkeypatch.setattr(sessions, "_with_db", unrelated_conflict)

    with pytest.raises(HTTPException) as exc:
        await sessions.backfill_session_owner_profiles(
            SessionOwnerBackfill(profile="work"), cast(Request, SimpleNamespace()))

    assert exc.value.status_code == 409
    assert exc.value.detail == "session_busy"
    assert attempts == 1
