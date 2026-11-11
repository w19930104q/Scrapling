"""Tests for per-session robots identities and per-(source, identity) robots pacing.

Covers:
- two sessions on the same crawler obeying different robots.txt groups,
- rules-cache isolation by source (scheme, host, effective port) and identity,
- one robots.txt fetch per group under concurrency, retry after cancellation,
- Crawl-delay / Request-rate group pacing on a fixed clock,
- stacking of robots pacing with download_delay and AutoThrottle,
- retries counting and cache hits not counting against the group limit.
"""

import asyncio

import pytest

from scrapling.engines.toolbelt.custom import Response
from scrapling.spiders.engine import CrawlerEngine
from scrapling.spiders.request import Request
from scrapling.spiders.robotstxt import RobotsGroupLimiter, RobotsTxtManager, source_key
from scrapling.spiders.session import SessionManager
from scrapling.core._types import Any, Dict, AsyncGenerator


# ---------------------------------------------------------------------------
# Local fixtures: robots.txt contents and canned responses (no network)
# ---------------------------------------------------------------------------

ROBOTS_TWO_IDENTITIES = """\
User-agent: alpha
Disallow: /alpha-only/
Crawl-delay: 5

User-agent: beta
Disallow: /beta-only/
Crawl-delay: 1
"""

ROBOTS_WILDCARD_DELAY = "User-agent: *\nCrawl-delay: 2\n"

ROBOTS_DELAY_AND_RATE = "User-agent: *\nCrawl-delay: 3\nRequest-rate: 2/10\n"


def make_response(url: str = "https://example.com", body: bytes = b"<html>page</html>", status: int = 200) -> Response:
    return Response(
        url=url,
        content=body,
        status=status,
        reason="OK",
        encoding="utf-8",
        cookies={},
        headers={"content-type": "text/html"},
        request_headers={"user-agent": "test"},
        method="GET",
    )


class FixtureSession:
    """Session stand-in that serves robots.txt and pages from local fixtures."""

    def __init__(self, robots_body: bytes = b"", clock=None):
        self._is_alive = False
        self._robots_body = robots_body
        self._clock = clock
        self.fetch_calls: list[str] = []
        self.sent_times: list[float] = []  # timestamps of page sends, when a clock is given

    async def __aenter__(self):
        self._is_alive = True
        return self

    async def __aexit__(self, *args):
        self._is_alive = False

    async def fetch(self, url: str, **kwargs):
        self.fetch_calls.append(url)
        if url.endswith("/robots.txt"):
            return make_response(url=url, body=self._robots_body)
        if self._clock is not None:
            self.sent_times.append(self._clock.time())
        return make_response(url=url)


class _LogCounterStub:
    def get_counts(self) -> Dict[str, int]:
        return {"debug": 0, "info": 0, "warning": 0, "error": 0, "critical": 0}


class MockSpider:
    """Minimal spider stub, mirroring the attributes CrawlerEngine reads."""

    def __init__(
        self,
        *,
        initial_requests: list[Request] | None = None,
        start_urls: list[str] | None = None,
        robots_txt_obey: bool = True,
        download_delay: float = 0.0,
        concurrent_requests: int = 4,
        concurrent_requests_per_domain: int = 0,
        max_blocked_retries: int = 3,
        autothrottle_enabled: bool = False,
        autothrottle_start_delay: float = 5.0,
        autothrottle_max_delay: float = 60.0,
        autothrottle_target_concurrency: float | None = None,
        autothrottle_block_backoff: bool = True,
        development_mode: bool = False,
        development_cache_dir: str | None = None,
        is_blocked_fn=None,
    ):
        self.name = "test_robots_identity"
        self.concurrent_requests = concurrent_requests
        self.concurrent_requests_per_domain = concurrent_requests_per_domain
        self.download_delay = download_delay
        self.max_blocked_retries = max_blocked_retries
        self.autothrottle_enabled = autothrottle_enabled
        self.autothrottle_start_delay = autothrottle_start_delay
        self.autothrottle_max_delay = autothrottle_max_delay
        self.autothrottle_target_concurrency = autothrottle_target_concurrency
        self.autothrottle_block_backoff = autothrottle_block_backoff
        self.allowed_domains: set[str] = set()
        self.fp_include_kwargs = False
        self.fp_include_headers = False
        self.fp_keep_fragments = False
        self.robots_txt_obey = robots_txt_obey
        self.development_mode = development_mode
        self.development_cache_dir = development_cache_dir
        self.start_urls = start_urls or []
        self._initial_requests = initial_requests or []
        self._is_blocked_fn = is_blocked_fn
        self._log_counter = _LogCounterStub()
        self.scraped_items: list[dict] = []

    async def start_requests(self) -> AsyncGenerator[Request, None]:
        for request in self._initial_requests:
            yield request

    async def parse(self, response) -> AsyncGenerator[Dict[str, Any] | Request | None, None]:
        yield {"url": str(response)}

    async def on_start(self, resuming: bool = False) -> None:
        pass

    async def on_close(self) -> None:
        pass

    async def on_error(self, request: Request, error: Exception) -> None:
        pass

    async def on_scraped_item(self, item: Dict[str, Any]) -> Dict[str, Any] | None:
        self.scraped_items.append(item)
        return item

    async def is_blocked(self, response) -> bool:
        return bool(self._is_blocked_fn and self._is_blocked_fn(response))

    async def retry_blocked_request(self, request: Request, response) -> Request:
        return request


class FakeClock:
    """Fixed clock for tests: time only moves when something sleeps."""

    def __init__(self):
        self.now = 0.0

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.now += seconds


class MockRobotsResponse:
    """Response stub for the robots.txt fetch callable."""

    def __init__(self, status: int = 200, body: bytes = b"", encoding: str = "utf-8"):
        self.status = status
        self.body = body
        self.encoding = encoding


def make_counting_fetch(content: str = "", status: int = 200):
    calls: list[tuple[str, str]] = []

    async def _fetch(url: str, sid: str) -> MockRobotsResponse:
        calls.append((url, sid))
        return MockRobotsResponse(status=status, body=content.encode())

    _fetch.calls = calls
    return _fetch


# ---------------------------------------------------------------------------
# source_key: effective-port normalization
# ---------------------------------------------------------------------------


class TestSourceKey:
    def test_default_ports_are_filled_in(self):
        assert source_key("http://example.com/a") == ("http", "example.com", 80)
        assert source_key("https://example.com/a") == ("https", "example.com", 443)

    def test_equivalent_default_port_urls_share_a_source(self):
        assert source_key("http://example.com/a") == source_key("http://example.com:80/b")
        assert source_key("https://example.com/a") == source_key("https://example.com:443/b?x=1")

    def test_non_default_port_is_kept(self):
        assert source_key("http://example.com:8080/a") == ("http", "example.com", 8080)

    def test_host_and_scheme_are_normalized(self):
        assert source_key("HTTP://Example.COM/a") == ("http", "example.com", 80)

    def test_different_schemes_or_ports_are_different_sources(self):
        assert source_key("http://example.com/") != source_key("https://example.com/")
        assert source_key("http://example.com:8000/") != source_key("http://example.com:9000/")


# ---------------------------------------------------------------------------
# RobotsTxtManager: per-identity rules
# ---------------------------------------------------------------------------


class TestPerIdentityRules:
    @pytest.mark.asyncio
    async def test_two_identities_get_different_allow_rules(self):
        mgr = RobotsTxtManager(make_counting_fetch(content=ROBOTS_TWO_IDENTITIES))

        assert await mgr.can_fetch("https://example.com/alpha-only/x", "s1", "alpha") is False
        assert await mgr.can_fetch("https://example.com/beta-only/x", "s1", "alpha") is True
        assert await mgr.can_fetch("https://example.com/beta-only/x", "s2", "beta") is False
        assert await mgr.can_fetch("https://example.com/alpha-only/x", "s2", "beta") is True

    @pytest.mark.asyncio
    async def test_two_identities_get_different_delay_directives(self):
        mgr = RobotsTxtManager(make_counting_fetch(content=ROBOTS_TWO_IDENTITIES))

        c_delay_a, _ = await mgr.get_delay_directives("https://example.com/", "s1", "alpha")
        c_delay_b, _ = await mgr.get_delay_directives("https://example.com/", "s2", "beta")
        c_delay_w, _ = await mgr.get_delay_directives("https://example.com/", "s3")

        assert c_delay_a == 5.0
        assert c_delay_b == 1.0
        assert c_delay_w is None  # no `*` group in this robots.txt

    @pytest.mark.asyncio
    async def test_unset_identity_falls_back_to_wildcard(self):
        mgr = RobotsTxtManager(make_counting_fetch(content="User-agent: *\nDisallow: /admin/"))

        assert await mgr.can_fetch("https://example.com/admin/", "s1") is False
        assert await mgr.can_fetch("https://example.com/admin/", "s1", "*") is False
        # Both calls used the same cache group
        assert ("https", "example.com", 443, "*") in mgr._cache


# ---------------------------------------------------------------------------
# RobotsTxtManager: cache isolation by source and identity
# ---------------------------------------------------------------------------


class TestCacheIsolation:
    @pytest.mark.asyncio
    async def test_equivalent_default_ports_share_one_fetch(self):
        fetch_fn = make_counting_fetch()
        mgr = RobotsTxtManager(fetch_fn)

        await mgr.can_fetch("http://example.com/a", "s1")
        await mgr.can_fetch("http://example.com:80/b", "s1")

        assert len(fetch_fn.calls) == 1

    @pytest.mark.asyncio
    async def test_scheme_host_port_and_identity_each_isolate_the_cache(self):
        fetch_fn = make_counting_fetch()
        mgr = RobotsTxtManager(fetch_fn)

        await mgr.can_fetch("http://example.com/", "s1")  # 1: http:80
        await mgr.can_fetch("https://example.com/", "s1")  # 2: other scheme
        await mgr.can_fetch("http://example.com:8080/", "s1")  # 3: other port
        await mgr.can_fetch("http://example.com/", "s1", "MyBot")  # 4: other identity
        await mgr.can_fetch("http://example.com/", "s1")  # cached, no fetch

        assert len(fetch_fn.calls) == 4

    @pytest.mark.asyncio
    async def test_concurrent_first_fetch_per_group_fetches_once(self):
        fetch_fn = make_counting_fetch(content=ROBOTS_TWO_IDENTITIES)

        async def slow_fetch(url: str, sid: str) -> MockRobotsResponse:
            await asyncio.sleep(0.01)
            return await fetch_fn(url, sid)

        mgr = RobotsTxtManager(slow_fetch)

        results = await asyncio.gather(
            *[mgr.can_fetch("https://example.com/alpha-only/", "s1", "alpha") for _ in range(4)],
            *[mgr.can_fetch("https://example.com/beta-only/", "s2", "beta") for _ in range(4)],
        )

        # One fetch per (source, identity) group, not per caller
        assert len(fetch_fn.calls) == 2
        assert results[:4] == [False] * 4
        assert results[4:] == [False] * 4


# ---------------------------------------------------------------------------
# RobotsTxtManager: cancellation leaves no permanent placeholder
# ---------------------------------------------------------------------------


class TestCancellation:
    @pytest.mark.asyncio
    async def test_cancelled_initiator_allows_retry(self):
        started = asyncio.Event()
        release = asyncio.Event()
        calls: list[str] = []

        async def hanging_fetch(url: str, sid: str) -> MockRobotsResponse:
            calls.append(url)
            started.set()
            await release.wait()
            return MockRobotsResponse(status=200, body=b"User-agent: *\nDisallow: /admin/")

        mgr = RobotsTxtManager(hanging_fetch)

        task = asyncio.create_task(mgr.can_fetch("https://example.com/", "s1"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert len(calls) == 1
        assert mgr._cache == {}  # nothing cached by the cancelled fetch

        # The group is not stuck: the next caller fetches again and succeeds
        release.set()
        assert await mgr.can_fetch("https://example.com/", "s1") is True
        assert await mgr.can_fetch("https://example.com/admin/", "s1") is False
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# RobotsGroupLimiter on a fixed clock
# ---------------------------------------------------------------------------


class TestRobotsGroupLimiter:
    @staticmethod
    async def _run_concurrent(limiter: RobotsGroupLimiter, clock: FakeClock, count: int) -> list[float]:
        """Run `count` concurrent acquires and return the send times they observed."""
        sends: list[float] = []

        async def acquire() -> None:
            await limiter.acquire()
            sends.append(clock.time())

        await asyncio.gather(*[acquire() for _ in range(count)])
        return sorted(sends)

    @pytest.mark.asyncio
    async def test_crawl_delay_spaces_concurrent_sends(self):
        clock = FakeClock()
        limiter = RobotsGroupLimiter(crawl_delay=2.0, clock=clock.time, sleep_fn=clock.sleep)

        sends = await self._run_concurrent(limiter, clock, 5)

        assert sends == [0.0, 2.0, 4.0, 6.0, 8.0]

    @pytest.mark.asyncio
    async def test_request_rate_caps_half_open_windows(self):
        clock = FakeClock()
        limiter = RobotsGroupLimiter(request_rate=(2, 10), clock=clock.time, sleep_fn=clock.sleep)

        sends = await self._run_concurrent(limiter, clock, 5)

        assert sends == [0.0, 0.0, 10.0, 10.0, 20.0]
        # No half-open 10s window holds more than 2 sends
        for send in sends:
            assert sum(1 for s in sends if send <= s < send + 10) <= 2

    @pytest.mark.asyncio
    async def test_crawl_delay_and_request_rate_stack(self):
        clock = FakeClock()
        limiter = RobotsGroupLimiter(crawl_delay=3.0, request_rate=(2, 10), clock=clock.time, sleep_fn=clock.sleep)

        sends = await self._run_concurrent(limiter, clock, 5)

        assert sends == [0.0, 3.0, 10.0, 13.0, 20.0]
        # Both constraints hold at every send
        for earlier, later in zip(sends, sends[1:]):
            assert later - earlier >= 3.0
        for send in sends:
            assert sum(1 for s in sends if send <= s < send + 10) <= 2

    @pytest.mark.asyncio
    async def test_groups_do_not_share_limits(self):
        clock_a, clock_b = FakeClock(), FakeClock()
        limiter_a = RobotsGroupLimiter(crawl_delay=5.0, clock=clock_a.time, sleep_fn=clock_a.sleep)
        limiter_b = RobotsGroupLimiter(crawl_delay=1.0, clock=clock_b.time, sleep_fn=clock_b.sleep)

        for _ in range(3):
            await limiter_a.acquire()
            await limiter_b.acquire()

        assert list(limiter_a._sent) == [0.0, 5.0, 10.0]
        assert list(limiter_b._sent) == [0.0, 1.0, 2.0]

    @pytest.mark.asyncio
    async def test_no_directives_never_sleeps(self):
        clock = FakeClock()
        limiter = RobotsGroupLimiter(clock=clock.time, sleep_fn=clock.sleep)

        for _ in range(10):
            await limiter.acquire()

        assert clock.now == 0.0


# ---------------------------------------------------------------------------
# SessionManager: robots identity configuration
# ---------------------------------------------------------------------------


class TestSessionRobotsIdentity:
    def test_identity_defaults_to_wildcard(self):
        sm = SessionManager()
        sm.add("a", FixtureSession())

        assert sm.robots_identity_for("a") == "*"
        assert sm.robots_identity_for("unregistered") == "*"

    def test_configured_identity_is_returned(self):
        sm = SessionManager()
        sm.add("a", FixtureSession(), robots_identity="MyBot")
        sm.add("b", FixtureSession())

        assert sm.robots_identity_for("a") == "MyBot"
        assert sm.robots_identity_for("b") == "*"

    def test_pop_drops_the_identity(self):
        sm = SessionManager()
        sm.add("a", FixtureSession(), robots_identity="MyBot")
        sm.pop("a")

        assert sm.robots_identity_for("a") == "*"


# ---------------------------------------------------------------------------
# Engine: two identities on one crawler obey different rules
# ---------------------------------------------------------------------------


class TestEnginePerIdentityRules:
    @pytest.mark.asyncio
    async def test_two_sessions_obey_their_own_robots_group(self):
        spider = MockSpider(
            start_urls=["https://example.com/"],
            initial_requests=[
                Request("https://example.com/alpha-only/1", sid="sa"),
                Request("https://example.com/beta-only/1", sid="sb"),
                Request("https://example.com/public/a", sid="sa"),
                Request("https://example.com/public/b", sid="sb"),
            ],
        )
        session_a = FixtureSession(ROBOTS_TWO_IDENTITIES.encode())
        session_b = FixtureSession(ROBOTS_TWO_IDENTITIES.encode())
        sm = SessionManager()
        sm.add("sa", session_a, robots_identity="alpha")
        sm.add("sb", session_b, robots_identity="beta")
        engine = CrawlerEngine(spider, sm, clock=FakeClock().time)

        stats = await engine.crawl()

        # Each session was only blocked by its own group's Disallow
        assert stats.robots_disallowed_count == 2
        assert "https://example.com/alpha-only/1" not in session_a.fetch_calls
        assert "https://example.com/beta-only/1" not in session_b.fetch_calls
        assert "https://example.com/public/a" in session_a.fetch_calls
        assert "https://example.com/public/b" in session_b.fetch_calls

        # robots.txt was fetched once per (source, identity) group, not per request
        robots_fetches = [url for url in session_a.fetch_calls + session_b.fetch_calls if url.endswith("/robots.txt")]
        assert len(robots_fetches) == 2

        # Each (source, identity) group got its own cached rules and limiter
        key_a = ("https", "example.com", 443, "alpha")
        key_b = ("https", "example.com", 443, "beta")
        assert set(engine._robots_rule_cache) == {key_a, key_b}
        floor_a, limiter_a = engine._robots_rule_cache[key_a]
        floor_b, limiter_b = engine._robots_rule_cache[key_b]
        assert floor_a == 5.0
        assert floor_b == 1.0
        assert limiter_a is not limiter_b

    @pytest.mark.asyncio
    async def test_rule_cache_isolated_by_source(self):
        spider = MockSpider()
        sm = SessionManager()
        sm.add("sa", FixtureSession(), robots_identity="alpha")
        engine = CrawlerEngine(spider, sm)

        await engine._get_robots_rules(Request("http://example.com/", sid="sa"))
        await engine._get_robots_rules(Request("http://example.com:80/other", sid="sa"))  # same source
        await engine._get_robots_rules(Request("https://example.com/", sid="sa"))  # other scheme
        await engine._get_robots_rules(Request("http://example.com:8080/", sid="sa"))  # other port

        assert set(engine._robots_rule_cache) == {
            ("http", "example.com", 80, "alpha"),
            ("https", "example.com", 443, "alpha"),
            ("http", "example.com", 8080, "alpha"),
        }


# ---------------------------------------------------------------------------
# Engine: group pacing on a fixed clock
# ---------------------------------------------------------------------------


class TestEngineGroupPacing:
    @pytest.mark.asyncio
    async def test_retries_count_against_the_group_limit(self):
        clock = FakeClock()
        spider = MockSpider(
            start_urls=["https://example.com/"],
            initial_requests=[Request("https://example.com/page", sid="default")],
            is_blocked_fn=lambda r: True,
            max_blocked_retries=3,
        )
        session = FixtureSession(ROBOTS_WILDCARD_DELAY.encode(), clock=clock)
        sm = SessionManager()
        sm.add("default", session)
        engine = CrawlerEngine(spider, sm, clock=clock.time, sleep_fn=clock.sleep)

        await engine.crawl()

        # The initial send plus 3 retries each went through the group limiter
        assert session.sent_times == [0.0, 2.0, 4.0, 6.0]

    @pytest.mark.asyncio
    async def test_cache_hits_do_not_count_against_the_group_limit(self, tmp_path):
        clock = FakeClock()
        spider = MockSpider(development_mode=True, development_cache_dir=str(tmp_path))
        session = FixtureSession(ROBOTS_WILDCARD_DELAY.encode(), clock=clock)
        sm = SessionManager()
        sm.add("default", session)
        engine = CrawlerEngine(spider, sm, clock=clock.time, sleep_fn=clock.sleep)

        request = Request("https://example.com/page", sid="default")
        request.update_fingerprint()
        await engine._process_request(request)  # real fetch, counts
        await engine._process_request(request)  # cache hit, must not count

        assert engine.stats.cache_hits == 1
        assert session.sent_times == [0.0]

    @pytest.mark.asyncio
    async def test_request_rate_window_is_enforced_during_crawl(self):
        clock = FakeClock()
        spider = MockSpider(
            start_urls=["https://example.com/"],
            initial_requests=[Request(f"https://example.com/p{i}", sid="default") for i in range(5)],
            concurrent_requests=8,  # high enough that only robots pacing shapes the sends
        )
        session = FixtureSession(ROBOTS_DELAY_AND_RATE.encode(), clock=clock)
        sm = SessionManager()
        sm.add("default", session)
        engine = CrawlerEngine(spider, sm, clock=clock.time, sleep_fn=clock.sleep)

        await engine.crawl()

        sends = sorted(session.sent_times)
        assert sends == [0.0, 3.0, 10.0, 13.0, 20.0]


# ---------------------------------------------------------------------------
# Engine: robots pacing stacks with download_delay and AutoThrottle
# ---------------------------------------------------------------------------


class TestStacking:
    @pytest.mark.asyncio
    async def test_floor_is_max_of_spider_delay_and_robots_directives(self):
        spider = MockSpider(download_delay=4.0)
        sm = SessionManager()
        sm.add("default", FixtureSession(ROBOTS_DELAY_AND_RATE.encode()))
        engine = CrawlerEngine(spider, sm)

        floor, limiter = await engine._get_robots_rules(Request("https://example.com/", sid="default"))

        # max(download_delay=4, Crawl-delay=3, Request-rate 10/2=5)
        assert floor == 5.0
        assert limiter._crawl_delay == 3.0
        assert limiter._request_rate == (2, 10)

    @pytest.mark.asyncio
    async def test_autothrottle_cannot_go_below_the_robots_floor(self):
        spider = MockSpider(
            autothrottle_enabled=True,
            autothrottle_start_delay=1.0,
            autothrottle_max_delay=60.0,
        )
        sm = SessionManager()
        sm.add("default", FixtureSession(ROBOTS_DELAY_AND_RATE.encode()))
        engine = CrawlerEngine(spider, sm)

        floor, _ = await engine._get_robots_rules(Request("https://example.com/", sid="default"))

        assert floor == 5.0
        assert engine._autothrottle.delay_for("example.com", floor) >= floor
        # A fast healthy response still can't pull the delay under the floor
        new_delay = engine._autothrottle.record("example.com", latency=0.01, ok=True, floor=floor)
        assert new_delay >= floor

    @pytest.mark.asyncio
    async def test_robots_disabled_behavior_is_unchanged(self):
        spider = MockSpider(robots_txt_obey=False, download_delay=1.5)
        session = FixtureSession(b"User-agent: *\nDisallow: /\n")
        sm = SessionManager()
        sm.add("default", session, robots_identity="alpha")
        engine = CrawlerEngine(spider, sm)

        assert engine._robots_manager is None
        floor, limiter = await engine._get_robots_rules(Request("https://example.com/", sid="default"))
        assert floor == 1.5
        assert limiter is None

        await engine._process_request(Request("https://example.com/page", sid="default"))

        # The disallow-all robots.txt was never even fetched
        assert engine.stats.requests_count == 1
        assert session.fetch_calls == ["https://example.com/page"]
