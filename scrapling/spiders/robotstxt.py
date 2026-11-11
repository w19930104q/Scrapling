from collections import deque
from urllib.parse import urlparse

from anyio import Lock, create_task_group, current_time, sleep
from protego import Protego

from scrapling.core._types import Dict, Tuple, Optional, Callable, Awaitable
from scrapling.core.utils import log

#: The robots.txt user-agent token used when a session has no identity configured.
WILDCARD_IDENTITY = "*"

_DEFAULT_PORTS = {"http": 80, "https": 443}

#: Cache key for one rules group: (scheme, host, effective port, robots identity)
GroupKey = Tuple[str, str, int, str]


def source_key(url: str) -> Tuple[str, str, int]:
    """Return the ``(scheme, host, port)`` origin tuple of a URL.

    The port is the effective one: when the URL doesn't state it, the scheme's
    default port is filled in, so ``http://example.com`` and
    ``http://example.com:80`` map to the same source.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port or _DEFAULT_PORTS.get(scheme, 80)
    return scheme, host, port


class RobotsTxtManager:
    """Manages fetching, parsing, and caching of robots.txt files.

    Parsed rules are cached per ``(scheme, host, effective port, identity)``
    group, so two sessions crawling the same site with different robots
    identities each get their own view of the rules.
    """

    def __init__(self, fetch_fn: Callable[[str, str], Awaitable]):
        self._fetch_fn = fetch_fn
        self._cache: Dict[GroupKey, Protego] = {}
        self._locks: Dict[GroupKey, Lock] = {}

    @staticmethod
    def _cache_key(url: str, identity: str) -> GroupKey:
        scheme, host, port = source_key(url)
        return scheme, host, port, identity or WILDCARD_IDENTITY

    async def _get_parser(self, url: str, sid: str, identity: str = WILDCARD_IDENTITY) -> Protego:
        key = self._cache_key(url, identity)

        if key in self._cache:
            return self._cache[key]

        async with self._locks.setdefault(key, Lock()):
            if key in self._cache:
                return self._cache[key]
            # If the task fetching here is cancelled, the lock is released and
            # nothing is cached, so the next caller simply retries the fetch.
            return await self._fetch_parser(key, sid)

    async def _fetch_parser(self, key: GroupKey, sid: str) -> Protego:
        """Fetch and parse the robots.txt of a group, then cache the parser."""
        scheme, host, port, _ = key
        default_port = _DEFAULT_PORTS.get(scheme, 80)
        netloc = host if port == default_port else f"{host}:{port}"
        robots_url = f"{scheme}://{netloc}/robots.txt"
        content = ""
        try:
            response = await self._fetch_fn(robots_url, sid)
            if response.status == 200:
                content = response.body.decode(response.encoding, errors="replace")
            elif 400 <= response.status < 500:
                content = ""
            else:
                log.warning(f"robots.txt for {netloc} returned status {response.status}; assuming full disallow")
                content = "User-agent: *\nDisallow: /"
        except Exception as e:
            log.warning(f"Failed to fetch robots.txt for {netloc}: {e}; assuming full disallow")
            content = "User-agent: *\nDisallow: /"

        try:
            # Left in place, a BOM hides the first line (usually `User-agent: *`) from protego
            parser = Protego.parse(content.removeprefix("\N{BYTE ORDER MARK}"))
        except Exception as e:
            log.warning(f"Failed to parse robots.txt for {netloc}: {e}")
            parser = Protego.parse("")

        self._cache[key] = parser
        return parser

    async def can_fetch(self, url: str, sid: str, identity: str = WILDCARD_IDENTITY) -> bool:
        """Check if a URL can be fetched according to the source's robots.txt.

        :param url: The full URL to check
        :param sid: Session ID for fetching robots.txt if not yet cached
        :param identity: The robots user-agent token whose rules apply; defaults to ``*``
        """
        parser = await self._get_parser(url, sid, identity)
        return parser.can_fetch(url, identity or WILDCARD_IDENTITY)

    async def get_delay_directives(
        self, url: str, sid: str, identity: str = WILDCARD_IDENTITY
    ) -> tuple[Optional[float], Optional[tuple[int, int]]]:
        """Return both crawl-delay and request-rate in a single parser lookup.

        :param url: Any URL on the source to check
        :param sid: Session ID for fetching robots.txt if not yet cached
        :param identity: The robots user-agent token whose rules apply; defaults to ``*``
        """
        parser = await self._get_parser(url, sid, identity)
        agent = identity or WILDCARD_IDENTITY
        c_delay = parser.crawl_delay(agent)
        rate = parser.request_rate(agent)
        return (
            float(c_delay) if c_delay is not None else None,
            (rate.requests, rate.seconds) if rate is not None else None,
        )

    async def prefetch(self, urls: list[str], sid: str, identities: tuple[str, ...] = (WILDCARD_IDENTITY,)) -> None:
        """Pre-warm the robots.txt cache for a list of seed URLs concurrently.

        :param urls: Seed URLs whose sources should be pre-fetched (one per source).
        :param sid: Session ID to use for the robots.txt fetch requests.
        :param identities: The robots identities to warm the cache for.
        """
        if not urls:
            return
        log.debug(f"Pre-fetching robots.txt for {len(urls)} source(s)")
        async with create_task_group() as tg:
            for url in urls:
                for identity in identities:
                    tg.start_soon(self._get_parser, url, sid, identity)


class RobotsGroupLimiter:
    """Serializes request starts of one ``(source, identity)`` group so they honor
    the group's robots.txt ``Crawl-delay`` and ``Request-rate`` directives.

    - ``Crawl-delay``: two consecutive sends of the group are at least this many
      seconds apart.
    - ``Request-rate: R/S``: any half-open window of ``S`` seconds contains at
      most ``R`` sends of the group.

    Both clocks are injectable so tests can drive the limiter with a fixed clock.
    """

    def __init__(
        self,
        crawl_delay: Optional[float] = None,
        request_rate: Optional[tuple[int, int]] = None,
        clock: Callable[[], float] = current_time,
        sleep_fn: Callable[[float], Awaitable] = sleep,
    ):
        """
        :param crawl_delay: Minimum seconds between two consecutive sends, or None.
        :param request_rate: A ``(requests, seconds)`` pair, or None.
        :param clock: Monotonic clock, defaults to `anyio.current_time`.
        :param sleep_fn: Sleep callable, defaults to `anyio.sleep`.
        """
        self._crawl_delay = crawl_delay or None
        self._request_rate = request_rate if request_rate and request_rate[0] > 0 else None
        self._sent: deque[float] = deque()
        self._clock = clock
        self._sleep = sleep_fn
        self._lock = Lock()

    def _prune(self, cutoff: float) -> None:
        """Drop sends at or before ``cutoff``; they fall outside any half-open window starting after it."""
        while self._sent and self._sent[0] <= cutoff:
            self._sent.popleft()

    async def acquire(self) -> None:
        """Wait until the group may send another request, then record the send."""
        async with self._lock:
            start = self._clock()

            if self._crawl_delay and self._sent:
                start = max(start, self._sent[-1] + self._crawl_delay)

            if self._request_rate:
                max_requests, period = self._request_rate
                self._prune(start - period)
                if len(self._sent) >= max_requests:
                    # The window is full; wait until the oldest send leaves it
                    start = max(start, self._sent[0] + period)
                    self._prune(start - period)

            wait = start - self._clock()
            if wait > 0:
                # Sleeping while holding the lock is what serializes the group
                await self._sleep(wait)

            self._sent.append(self._clock())
