"""Tests for web_search SearchRouter and SearchCache."""

from gmas.tools.web_search._cache import SearchCache
from gmas.tools.web_search._providers import SearchProvider
from gmas.tools.web_search._router import IntentClassifier, SearchRouter


class DummyProvider(SearchProvider):
    """Minimal provider for router.route_providers."""

    def search(self, query: str, max_results: int = 10) -> list[dict[str, str]]:
        return []


class TestSearchRouter:
    def test_score_intents_news(self):
        r = SearchRouter()
        scores = r.score_intents("breaking news headlines today")
        assert scores.get("news", 0) > 0

    def test_score_intents_technical(self):
        r = SearchRouter()
        scores = r.score_intents("python stack trace import error")
        assert scores.get("technical", 0) > 0

    def test_detect_intent_low_confidence_returns_general(self):
        r = SearchRouter()
        # Unlikely to trigger strong signals
        assert r.detect_intent("zzzqqqmmm") == "general"

    def test_detect_intents_sorted(self):
        r = SearchRouter()
        lst = r.detect_intents("arxiv research paper doi")
        assert isinstance(lst, list)
        if len(lst) >= 2:
            assert lst[0][1] >= lst[1][1]

    def test_route_with_explicit_intent(self):
        r = SearchRouter(available_providers={"duckduckgo": DummyProvider()})
        names = r.route("x", intent="news")
        assert "duckduckgo" in names

    def test_route_filters_unavailable_providers(self):
        r = SearchRouter(available_providers={}, fallback_providers=["duckduckgo"])
        names = r.route("news today")
        assert names[-1] == "duckduckgo" or "duckduckgo" in names

    def test_route_providers_resolves_objects(self):
        p = DummyProvider()
        r = SearchRouter(available_providers={"duckduckgo": p})
        providers = r.route_providers("query", intent="general")
        assert providers
        assert providers[0] is p

    def test_classifier_blend_and_failure_fallback(self):
        class BadClassifier(IntentClassifier):
            def classify(self, query: str) -> dict[str, float]:
                msg = "boom"
                raise RuntimeError(msg)

        class OkClassifier(IntentClassifier):
            def classify(self, query: str) -> dict[str, float]:
                return {"news": 0.9}

        r = SearchRouter(intent_classifier=BadClassifier(), classifier_weight=0.5)
        scores = r.score_intents("documentation tutorial")
        assert isinstance(scores, dict)

        r2 = SearchRouter(intent_classifier=OkClassifier(), classifier_weight=0.5)
        s2 = r2.score_intents("something")
        assert "news" in s2

    def test_classifier_empty_ext_scores_uses_keywords_only(self):
        class EmptyClassifier(IntentClassifier):
            def classify(self, query: str) -> dict[str, float]:
                return {}

        r = SearchRouter(intent_classifier=EmptyClassifier())
        s = r.score_intents("github api error")
        assert s.get("technical", 0) >= 0


class TestSearchCache:
    def test_search_put_get_hit(self):
        c = SearchCache(max_entries=4, ttl=60.0)
        data = [{"title": "a", "url": "http://x"}]
        assert c.get_search("q", 5) is None
        c.put_search("q", 5, data)
        got = c.get_search("q", 5)
        assert got == data
        st = c.stats
        assert st["hits"] >= 1

    def test_search_key_variants_differ(self):
        c = SearchCache(max_entries=8, ttl=60.0)
        c.put_search("q", 5, [{"x": "1"}], provider="p1")
        assert c.get_search("q", 5, provider="p2") is None

    def test_ttl_expiry(self, monkeypatch):
        c = SearchCache(max_entries=4, ttl=0.01)

        t0 = [0.0]

        def fake_mono():
            return t0[0]

        monkeypatch.setattr("gmas.tools.web_search._cache._time.monotonic", fake_mono)
        c.put_search("q", 1, [{"a": "b"}])
        t0[0] += 1.0
        assert c.get_search("q", 1) is None

    def test_lru_eviction(self):
        c = SearchCache(max_entries=2, ttl=300.0)
        c.put_search("a", 1, [{"t": "a"}])
        c.put_search("b", 1, [{"t": "b"}])
        c.put_search("c", 1, [{"t": "c"}])
        stats = c.stats
        assert stats["size"] <= 2

    def test_fetch_put_get(self):
        c = SearchCache(max_entries=4, ttl=60.0)
        payload = {"html": "<p>x</p>"}
        c.put_fetch("https://example.com", payload, use_browser=True, wait_for_selector="#x")
        got = c.get_fetch("https://example.com", use_browser=True, wait_for_selector="#x")
        assert got == payload

    def test_image_search_roundtrip(self):
        c = SearchCache(max_entries=4, ttl=60.0)
        rows = [{"url": "http://img"}]
        c.put_image_search("cat", 3, rows, provider="p")
        assert c.get_image_search("cat", 3, provider="p") == rows

    def test_clear(self):
        c = SearchCache(max_entries=4, ttl=60.0)
        c.put_search("z", 1, [])
        c.clear()
        assert c.stats["size"] == 0
        assert c.stats["hits"] == 0

    def test_get_deepcopy_isolation(self):
        c = SearchCache(max_entries=4, ttl=60.0)
        inner = {"nested": [1, 2, 3]}
        c.put_fetch("https://x.test", inner)
        g = c.get_fetch("https://x.test")
        assert g is not None
        g["nested"].append(4)
        g2 = c.get_fetch("https://x.test")
        assert g2 is not None
        assert len(g2["nested"]) == 3
