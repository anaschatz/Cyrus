from __future__ import annotations

import asyncio
import gzip
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401

import httpx

from jarvis_research.adapters import DirectHTTPFetcher, SearXNGSearchAdapter
from jarvis_research.extraction import extract_main_content
from jarvis_orchestrator.research import DateProvenance
from jarvis_orchestrator.research_adapter import LoopbackResearchAdapter


class TrackedStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes, *, delay: float = 0) -> None:
        self.body = body
        self.delay = delay
        self.closed = False

    async def __aiter__(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        yield self.body

    async def aclose(self):
        self.closed = True


class ResearchAdapterTests(unittest.TestCase):
    def test_loopback_adapter_rejects_deceptive_authorities_and_extra_url_parts(self) -> None:
        for url in (
            "http://127.0.0.1:8770@example.com/v1",
            "http://127.0.0.1:8770/extra/v1",
            "http://127.0.0.1:8770/v1?redirect=/v1",
            "http://127.0.0.1:8770/v1#fragment/v1",
            "http://127.0.0.1:99999/v1",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                LoopbackResearchAdapter(base_url=url, bearer_token="test-token")

    def test_cached_robots_rules_are_evaluated_for_every_path_and_redirect(self) -> None:
        async def run():
            requests = []

            async def handler(request):
                requests.append(request.url.path)
                if request.url.path == "/robots.txt":
                    return httpx.Response(200, content=b"User-agent: *\nDisallow: /private\n")
                if request.url.path == "/redirect":
                    return httpx.Response(302, headers={"location": "/private/redirected"})
                return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"public")

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                fetcher = DirectHTTPFetcher(client)
                await fetcher.fetch("https://example.com/public")
                for path in ("/private/direct", "/redirect"):
                    with self.assertRaisesRegex(RuntimeError, "robots"):
                        await fetcher.fetch("https://example.com" + path)
            self.assertFalse(any(path.startswith("/private") for path in requests))
            self.assertEqual(requests.count("/robots.txt"), 1)

        asyncio.run(run())

    def test_streamed_gzip_robots_are_decoded_before_applying_rules(self) -> None:
        async def run():
            stream = TrackedStream(gzip.compress(b"User-agent: *\nDisallow: /private\n"))

            async def handler(request):
                self.assertEqual(request.url.path, "/robots.txt")
                return httpx.Response(200, headers={"content-encoding": "gzip"}, stream=stream)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(RuntimeError, "^robots_disallowed$"):
                    await DirectHTTPFetcher(client).fetch("https://example.com/private")
            self.assertTrue(stream.closed)

        asyncio.run(run())

    def test_robots_body_obeys_its_deadline_and_closes_the_response(self) -> None:
        async def run():
            stream = TrackedStream(b"User-agent: *\nAllow: /\n", delay=1)

            async def handler(request):
                self.assertEqual(request.url.path, "/robots.txt")
                return httpx.Response(200, stream=stream)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(RuntimeError, "robots_failed_closed"):
                    await asyncio.wait_for(DirectHTTPFetcher(client).fetch("https://example.com/public"), 0.2)
            self.assertTrue(stream.closed)

        with patch("jarvis_research.adapters.ROBOTS_DEADLINE_SECONDS", 0.01, create=True):
            asyncio.run(run())

    def test_truncated_compressed_document_is_rejected_and_stream_is_closed(self) -> None:
        async def run():
            stream = TrackedStream(gzip.compress(b"A sufficiently long research document.")[:-8])

            async def handler(request):
                if request.url.path == "/robots.txt":
                    return httpx.Response(404)
                return httpx.Response(200, headers={"content-type": "text/plain", "content-encoding": "gzip"}, stream=stream)

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(RuntimeError, "compressed_body_incomplete"):
                    await DirectHTTPFetcher(client).fetch("https://example.com/public")
            self.assertTrue(stream.closed)

        asyncio.run(run())

    def test_public_ipv6_origin_remains_a_valid_url(self) -> None:
        self.assertEqual(
            DirectHTTPFetcher._origin("https://[2606:4700:4700::1111]:443/article"),
            "https://[2606:4700:4700::1111]:443",
        )

    def test_main_content_is_deterministic_and_metadata_dates_are_rule_bound(self) -> None:
        body = b"""<html><head><title>Ignored fallback</title>
        <meta charset='utf-8'><meta http-equiv='content-type' content='text/html'>
        <meta><meta name=''>
        <meta property='og:title' content='Official title'>
        <meta property='og:site_name' content='Publisher'>
        <meta property='article:published_time' content='2026-08-10T10:00:00+03:00'>
        <link rel='canonical' href='https://example.com/canonical'></head>
        <body><nav><p>menu must be excluded completely</p></nav>
        <main><h1>Official title with enough content</h1>
        <p>The exact factual sentence remains in normalized storage for quote-first locators.</p></main>
        <script>prompt injection must not be evidence</script></body></html>"""
        extracted = extract_main_content(body, "text/html")
        self.assertEqual(extracted.title, "Official title")
        self.assertEqual(extracted.publisher, "Publisher")
        self.assertEqual(extracted.canonical_url, "https://example.com/canonical")
        self.assertEqual(extracted.date_provenance, DateProvenance.ARTICLE_PUBLISHED_TIME)
        self.assertNotIn("menu", extracted.document.text)
        self.assertNotIn("prompt injection", extracted.document.text)
        self.assertIn("quote-first locators", extracted.document.text)

    def test_inline_markup_preserves_words_punctuation_and_line_breaks(self) -> None:
        body = b"<p>This inter<strong>national</strong> report is <em>verified</em>.<br>Read it carefully.</p>"
        extracted = extract_main_content(body, "text/html")
        self.assertEqual(extracted.document.text, "This international report is verified. Read it carefully.")

    def test_literal_entity_text_is_decoded_once_for_html_and_preserved_in_plain_text(self) -> None:
        expected = "This report explains the literal entity &amp; and preserves the original evidence."
        for body, mime in (
            (b"<p>This report explains the literal entity &amp;amp; and preserves the original evidence.</p>", "text/html"),
            (expected.encode(), "text/plain"),
        ):
            with self.subTest(mime=mime):
                extracted = extract_main_content(body, mime)
                self.assertEqual(extracted.document.text, expected)

    def test_nested_blocks_and_unclosed_last_paragraph_keep_all_evidence(self) -> None:
        body = (
            b"<blockquote>Important introductory evidence."
            b"<p>Inner paragraph contains detailed evidence.</p>"
            b"Important final evidence.</blockquote>"
            b"<p>The final paragraph omits its optional closing tag."
        )
        extracted = extract_main_content(body, "text/html")
        self.assertEqual(extracted.document.text, "\n\n".join((
            "Important introductory evidence.", "Inner paragraph contains detailed evidence.",
            "Important final evidence.", "The final paragraph omits its optional closing tag.",
        )))
        self.assertEqual(extracted.block_count, 4)

    def test_searxng_uses_fixed_internal_post_json_and_dedupes_urls(self) -> None:
        captured = []

        async def handler(request):
            captured.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "results": [
                        {"url": "https://example.com/a", "title": "A", "content": "one", "engine": "official"},
                        {"url": "https://example.com/a", "title": "duplicate", "content": "two", "engine": "other"},
                        {"url": "https://example.org/b", "title": "B", "content": "three", "engine": "other"},
                    ]
                },
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

        async def run():
            try:
                return await SearXNGSearchAdapter(client).search("price", language="en", limit=8)
            finally:
                await client.aclose()

        result = asyncio.run(run())
        self.assertEqual([item.requested_url for item in result], ["https://example.com/a", "https://example.org/b"])
        self.assertEqual((captured[0].method, captured[0].url.host, captured[0].url.path), ("POST", "searxng", "/search"))
        form = captured[0].content.decode()
        self.assertIn("format=json", form)
        self.assertIn("safesearch=1", form)

    def test_direct_fetch_checks_robots_and_follows_bounded_redirect(self) -> None:
        requests = []

        async def handler(request):
            requests.append(str(request.url))
            if request.url.path == "/robots.txt":
                return httpx.Response(200, content=b"User-agent: *\nAllow: /\n")
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/final"})
            return httpx.Response(200, headers={"content-type": "text/html"}, content=b"<p>usable body</p>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False, trust_env=False)

        async def run():
            try:
                return await DirectHTTPFetcher(client).fetch("https://example.com/start")
            finally:
                await client.aclose()

        result = asyncio.run(run())
        self.assertEqual(result.final_url, "https://example.com/final")
        self.assertEqual(result.redirect_count, 1)
        self.assertEqual(requests.count("https://example.com/robots.txt"), 1)

    def test_robots_disallow_and_mime_fail_closed_without_document_body(self) -> None:
        async def disallow(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(200, content=b"User-agent: *\nDisallow: /private\n")
            raise AssertionError("document must not be fetched")

        async def bad_mime(request):
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"pdf")

        async def run(handler, url):
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
            try:
                return await DirectHTTPFetcher(client).fetch(url)
            finally:
                await client.aclose()

        with self.assertRaisesRegex(RuntimeError, "robots_disallowed"):
            asyncio.run(run(disallow, "https://example.com/private/file"))
        with self.assertRaisesRegex(RuntimeError, "mime_forbidden"):
            asyncio.run(run(bad_mime, "https://example.com/file"))

    def test_one_transient_retry_token_is_shared_with_robots(self) -> None:
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(503)
            if request.url.path == "/robots.txt":
                return httpx.Response(404)
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"ok")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)

        async def run():
            try:
                return await DirectHTTPFetcher(client).fetch("https://example.com/file")
            finally:
                await client.aclose()

        result = asyncio.run(run())
        self.assertEqual(result.body, b"ok")
        self.assertEqual(calls, 3)


if __name__ == "__main__":
    unittest.main()
