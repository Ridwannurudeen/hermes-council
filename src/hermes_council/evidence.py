"""Evidence retrieval for council deliberation."""

from __future__ import annotations

import html
import http.client
import ipaddress
import os
import re
import socket
import ssl
import urllib.parse
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Iterable


_URL_RE = re.compile(r"https?://[^\s\)\]\"'<>]+")
_MAX_REDIRECTS = 5
_MAX_BODY_BYTES = 128_000
_USER_AGENT = "hermes-council/0.1 (+https://github.com/Ridwannurudeen/hermes-council)"


@dataclass(frozen=True)
class EvidenceSource:
    """A retrieved source or search result passed into the council."""

    title: str
    url: str
    snippet: str
    source_type: str
    verified: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceBundle:
    """Evidence retrieval result plus non-fatal retrieval errors."""

    sources: list[EvidenceSource]
    errors: list[str]

    @property
    def verified_sources(self) -> list[EvidenceSource]:
        return [source for source in self.sources if source.verified]

    def to_prompt_block(self) -> str:
        if not self.sources:
            return (
                "Evidence retrieval ran, but no external sources were retrieved. "
                "Treat uncited factual claims as uncertain."
            )

        lines = [
            "Evidence retrieved by hermes-council. Treat verified entries as external evidence; "
            "treat unverified-search-result entries as leads only.",
        ]
        for index, source in enumerate(self.sources, start=1):
            status = "verified" if source.verified else "unverified-search-result"
            lines.append(
                f"[{index}] {source.title} ({status})\n"
                f"URL: {source.url}\n"
                f"Snippet: {source.snippet}"
            )
        return "\n\n".join(lines)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        text = data.strip()
        if not text or self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(text)
        self.parts.append(text)


def extract_urls(text: str) -> list[str]:
    """Extract unique HTTP(S) URLs from text in first-seen order."""
    urls = []
    for match in _URL_RE.findall(text or ""):
        cleaned = match.rstrip(".,;:")
        if cleaned not in urls:
            urls.append(cleaned)
    return urls


def _strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _is_non_public(address: ipaddress._BaseAddress) -> bool:
    return (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _validate_public_url(url: str) -> tuple[str, str, int, str]:
    """Validate scheme + host shape. Return (scheme, host, port, path_with_query).

    Rejects literal private IPs and obfuscated numeric forms (decimal-encoded
    IPv4, 0x-prefixed hex, all-numeric-with-dots) before any network call.
    Hostname-to-IP validation happens at connect time in the connection
    classes below — this function alone is not sufficient to prevent SSRF.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("only http and https URLs are supported")
    if not parsed.hostname:
        raise ValueError("URL host is required")

    host = parsed.hostname.strip("[]").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("local hostnames are not allowed")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host.startswith("0x"):
            raise ValueError("obfuscated IP encoding is not allowed") from None
        if host and all(c.isdigit() or c == "." for c in host):
            raise ValueError("malformed numeric host is not allowed") from None
    else:
        if _is_non_public(address):
            raise ValueError("private or non-routable IP addresses are not allowed")

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.scheme, host, port, path


def _resolve_public_addresses(host: str) -> list[tuple[int, str]]:
    """Resolve host and reject if any returned address is non-public.

    Returns list of (family, ip) for the resolved addresses. Raises ValueError
    if resolution fails or any address is private/loopback/link-local/etc.
    Strict-on-any-private semantics defeat DNS rebinding round-robin tricks.
    """
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"DNS resolution failed: {exc}") from exc

    addresses: list[tuple[int, str]] = []
    for family, _, _, _, sockaddr in infos:
        ip = sockaddr[0]
        try:
            parsed_ip = ipaddress.ip_address(ip)
        except ValueError:
            raise ValueError(f"resolver returned non-IP address: {ip}") from None
        if _is_non_public(parsed_ip):
            raise ValueError(f"host resolves to non-public address: {ip}")
        addresses.append((family, ip))

    if not addresses:
        raise ValueError("DNS returned no addresses")
    return addresses


class _ValidatedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection that pins the connect target to a validated public IP."""

    def connect(self) -> None:
        family, ip = _resolve_public_addresses(self.host)[0]
        sock = socket.create_connection(
            (ip, self.port),
            timeout=self.timeout,
            source_address=self.source_address,
        )
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _ValidatedHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection that pins the connect target to a validated public IP."""

    def connect(self) -> None:
        family, ip = _resolve_public_addresses(self.host)[0]
        self.sock = socket.create_connection(
            (ip, self.port),
            timeout=self.timeout,
            source_address=self.source_address,
        )


def _read_url(url: str, timeout: float) -> str:
    """Fetch a URL, following redirects with per-hop validation.

    Each hop revalidates the URL against `_validate_public_url` and forces
    DNS resolution through `_resolve_public_addresses`, so a 30x to a
    private IP cannot bypass the SSRF guard. Caps at `_MAX_REDIRECTS`.
    """
    visited: set[str] = set()
    current = url

    for _ in range(_MAX_REDIRECTS + 1):
        if current in visited:
            raise ValueError("redirect loop detected")
        visited.add(current)

        scheme, host, port, path = _validate_public_url(current)
        conn_cls = _ValidatedHTTPSConnection if scheme == "https" else _ValidatedHTTPConnection
        if scheme == "https":
            conn = conn_cls(host, port, timeout=timeout, context=ssl.create_default_context())
        else:
            conn = conn_cls(host, port, timeout=timeout)

        try:
            conn.request("GET", path, headers={"User-Agent": _USER_AGENT})
            response = conn.getresponse()

            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError(
                        f"redirect status {response.status} with no Location header"
                    )
                current = urllib.parse.urljoin(current, location)
                continue

            if response.status >= 400:
                raise OSError(f"HTTP {response.status}")

            charset = response.headers.get_content_charset() or "utf-8"
            raw = response.read(_MAX_BODY_BYTES)
            return raw.decode(charset, errors="replace")
        finally:
            conn.close()

    raise ValueError("too many redirects")


def _source_from_html(url: str, raw_html: str, source_type: str) -> EvidenceSource:
    parser = _TextExtractor()
    parser.feed(raw_html)
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
    body = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
    if not title:
        title = urllib.parse.urlparse(url).netloc or url
    snippet = body[:700] if body else title
    return EvidenceSource(
        title=title[:180],
        url=url,
        snippet=snippet,
        source_type=source_type,
        verified=True,
    )


def _normalize_duckduckgo_url(url: str) -> str:
    parsed = urllib.parse.urlparse(html.unescape(url))
    if (
        parsed.path.startswith("/l/")
        and (not parsed.netloc or parsed.netloc.endswith("duckduckgo.com"))
    ):
        params = urllib.parse.parse_qs(parsed.query)
        if params.get("uddg"):
            return params["uddg"][0]
    return html.unescape(url)


def _parse_duckduckgo_results(raw_html: str, max_results: int) -> list[EvidenceSource]:
    links = re.findall(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    snippets = re.findall(
        r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>|'
        r'<div[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</div>',
        raw_html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    cleaned_snippets = []
    for first, second in snippets:
        cleaned_snippets.append(_strip_tags(first or second))

    results = []
    for index, (url, title) in enumerate(links):
        normalized = _normalize_duckduckgo_url(url)
        if not normalized.startswith(("http://", "https://")):
            continue
        snippet = cleaned_snippets[index] if index < len(cleaned_snippets) else ""
        results.append(
            EvidenceSource(
                title=_strip_tags(title)[:180] or normalized,
                url=normalized,
                snippet=snippet[:700] or "Search result returned without a snippet.",
                source_type="search_result",
                verified=False,
            )
        )
        if len(results) >= max_results:
            break
    return results


def _fetch_source(url: str, timeout: float, source_type: str) -> EvidenceSource:
    _validate_public_url(url)
    return _source_from_html(url, _read_url(url, timeout), source_type)


def _search_web(query: str, max_results: int, timeout: float) -> list[EvidenceSource]:
    url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    return _parse_duckduckgo_results(_read_url(url, timeout), max_results)


def _dedupe_sources(sources: Iterable[EvidenceSource]) -> list[EvidenceSource]:
    deduped = []
    seen = set()
    for source in sources:
        key = source.url.rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def collect_evidence(
    question: str,
    context: str = "",
    *,
    enabled: bool = True,
    max_sources: int = 5,
    timeout: float | None = None,
) -> EvidenceBundle:
    """Retrieve URL and search evidence for a council question.

    The retriever first verifies URLs supplied in the question/context. If more
    evidence is needed, it queries DuckDuckGo's HTML endpoint and attempts to
    fetch the result pages. Search results that cannot be fetched are retained
    as unverified search snippets so callers can distinguish them.
    """
    if not enabled:
        return EvidenceBundle([], [])

    timeout = timeout if timeout is not None else float(os.getenv("COUNCIL_EVIDENCE_TIMEOUT", "8"))
    max_sources = max(0, max_sources)
    if max_sources == 0:
        return EvidenceBundle([], [])

    errors: list[str] = []
    sources: list[EvidenceSource] = []

    for url in extract_urls(f"{question}\n{context}"):
        if len(sources) >= max_sources:
            break
        try:
            sources.append(_fetch_source(url, timeout, "provided_url"))
        except (OSError, socket.timeout, TimeoutError, ValueError) as exc:
            errors.append(f"{url}: {exc}")

    if len(sources) < max_sources and os.getenv("COUNCIL_EVIDENCE_SEARCH", "1") != "0":
        try:
            candidates = _search_web(question, max_sources - len(sources), timeout)
        except (OSError, socket.timeout, TimeoutError, ValueError) as exc:
            candidates = []
            errors.append(f"search: {exc}")

        for candidate in candidates:
            if len(sources) >= max_sources:
                break
            try:
                sources.append(_fetch_source(candidate.url, timeout, "search_result"))
            except (OSError, socket.timeout, TimeoutError, ValueError) as exc:
                errors.append(f"{candidate.url}: {exc}")
                sources.append(candidate)

    return EvidenceBundle(_dedupe_sources(sources)[:max_sources], errors)
