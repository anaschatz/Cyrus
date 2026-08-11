from __future__ import annotations

import asyncio
import json
import unittest

import _bootstrap  # noqa: F401

import httpx

from jarvis_research.adapters import DirectHTTPFetcher, SearXNGSearchAdapter
from jarvis_research.extraction import extract_main_content
from jarvis_orchestrator.research import DateProvenance


class ResearchAdapterTests(unittest.TestCase):
    def test_main_content_is_deterministic_and_metadata_dates_are_rule_bound(self) -> None:
        body = b"""<html><head><title>Ignored fallback</title>
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
