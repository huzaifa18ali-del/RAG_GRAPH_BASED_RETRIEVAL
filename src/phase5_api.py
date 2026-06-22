#!/usr/bin/env python3
"""
phase5_api.py

Phase 5 of the NLP pipeline: Interactive Semantic Query / Q&A Interface.

Enhanced with:
  1. Natural question understanding (summary, comparison, quote, theme, emotion)
  2. Smart answer generation (coherent paragraphs, deduplication, glue text)
  3. Theme & emotion analysis (sentiment detection, theme tagging)
  4. Cross-reference & insight extraction (repeated ideas, cause/effect, bridges)
  5. Polished presentation (sections: Summary, Top Quotes, Analysis)
  6. Export results to .txt / .json
  7. FAISS index persistence, batch mode, hybrid scoring, centroid pre-filter
  8. GPU support for FAISS and sentence-transformers

Usage:
    python phase5_query.py                                  # interactive (exact)
    python phase5_query.py --backend faiss --persist_index  # FAISS + save index
    python phase5_query.py --batch_file queries.txt         # batch mode
    python phase5_query.py --rerank --alpha 0.4             # hybrid scoring
    python phase5_query.py --approx_clusters 3              # centroid pre-filter
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.preprocessing import normalize

# --- Sentence Transformers ---
from sentence_transformers import SentenceTransformer

# --- Sentiment Analysis ---
# Primary: transformer-based model (understands literary/contextual language)
# Fallback chain: TextBlob → VADER → keyword-based
try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not installed — falling back to keyword sentiment.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# --- FAISS ---
try:
    import faiss
    try:
        _FAISS_RES = faiss.StandardGpuResources()
        FAISS_GPU = True
    except AttributeError:
        _FAISS_RES = None
        FAISS_GPU = False
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU = False
    _FAISS_RES = None

# --- Cross-encoder ---
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    stream=sys.stderr,
)
log = logging.getLogger("phase5")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDINGS_PATH = os.path.join("data", "embeddings.npy")
SENTENCES_PATH = os.path.join("data", "output_v2.json")
IDEA_GRAPH_PATH = os.path.join("data", "idea_graph.json")
CLUSTER_PATH = os.path.join("data", "cluster_summary.json")
MANIFEST_PATH = os.path.join("data", "embeddings_manifest.json")
FAISS_INDEX_PATH = os.path.join("data", "faiss.index")
BATCH_OUTPUT_PATH = os.path.join("data", "batch_results.jsonl")
EXPORT_DIR = os.path.join("data", "exports")

EMBED_MODEL_NAME = "all-mpnet-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K = 5
FAISS_NPROBE = 8
FAISS_NLIST = 64
RERANK_FETCH_MULT = 4
DEFAULT_ALPHA = 0.5
DEFAULT_APPROX_K = 0
CONSOLE_WIDTH = 96


# ---------------------------------------------------------------------------
# Theme dictionary — keyword clusters for thematic tagging
# ---------------------------------------------------------------------------
THEME_DICTIONARY: dict[str, list[str]] = {
    "loneliness": [
        "lonely", "alone", "solitude", "solitary", "isolated", "isolation",
        "forsaken", "desolate", "abandoned", "forlorn", "secluded", "withdrawn",
        "companionless", "friendless", "estranged", "detached", "empty",
    ],
    "love": [
        "love", "loved", "loving", "beloved", "affection", "passion",
        "romantic", "romance", "desire", "adore", "devotion", "tender",
        "heart", "embrace", "kiss", "fondness", "infatuation", "cherish",
    ],
    "dreams": [
        "dream", "dreams", "dreaming", "dreamer", "fantasy", "imagine",
        "imagination", "vision", "visionary", "illusion", "reverie",
        "aspiration", "hope", "wished", "longing", "yearning",
    ],
    "regret": [
        "regret", "remorse", "sorry", "guilt", "shame", "mistake",
        "lament", "repent", "rue", "wistful", "hindsight", "sorrow",
        "apologize", "forgive", "forgiveness", "blame", "reproach",
    ],
    "hope": [
        "hope", "hopeful", "optimism", "optimistic", "bright", "promise",
        "faith", "believe", "trust", "future", "tomorrow", "dawn",
        "possibility", "chance", "renewal", "beginning", "light",
    ],
    "despair": [
        "despair", "hopeless", "hopelessness", "misery", "anguish",
        "suffering", "torment", "grief", "mourn", "weep", "tears",
        "darkness", "bleak", "gloomy", "gloom", "melancholy", "wretched",
    ],
    "nature": [
        "river", "sky", "stars", "moon", "sun", "night", "morning",
        "wind", "rain", "snow", "tree", "flower", "garden", "sea",
        "earth", "cloud", "fog", "mist", "spring", "winter", "autumn",
    ],
    "time": [
        "time", "moment", "memory", "memories", "past", "present",
        "future", "yesterday", "today", "forever", "eternity", "fleeting",
        "passage", "years", "days", "nights", "hours", "clock", "aging",
    ],
    "identity": [
        "self", "identity", "who am i", "soul", "spirit", "character",
        "person", "being", "existence", "purpose", "meaning", "life",
        "consciousness", "aware", "awakening", "transformation",
    ],
    "society": [
        "society", "people", "crowd", "city", "street", "world",
        "civilization", "culture", "community", "neighbor", "stranger",
        "public", "social", "class", "poverty", "wealth", "power",
    ],
}

# Build a reverse lookup: word → set of theme names
_WORD_TO_THEMES: dict[str, set[str]] = defaultdict(set)
for _theme, _words in THEME_DICTIONARY.items():
    for _w in _words:
        _WORD_TO_THEMES[_w.lower()].add(_theme)


# ---------------------------------------------------------------------------
# Emotion labels for sentiment classification
# ---------------------------------------------------------------------------
EMOTION_LABELS = {
    "very_negative":      (-1.0,  -0.6),
    "negative":           (-0.6,  -0.2),
    "slightly_negative":  (-0.2,  -0.05),
    "neutral":            (-0.05,  0.05),
    "slightly_positive":  ( 0.05,  0.2),
    "positive":           ( 0.2,   0.6),
    "very_positive":      ( 0.6,   1.01),   # 1.01 so exact 1.0 is captured
}


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------
class Timer:
    _registry: dict[str, list[float]] = {}

    def __init__(self, stage: str, log_level: int = logging.DEBUG) -> None:
        self.stage = stage
        self.log_level = log_level
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1_000
        Timer._registry.setdefault(self.stage, []).append(self.elapsed_ms)
        log.log(self.log_level, "  %-28s %7.2f ms", self.stage, self.elapsed_ms)

    @classmethod
    def report(cls) -> str:
        if not cls._registry:
            return "  (no timing data)"
        lines = ["", "  ── Timing Report ──────────────────────────────────"]
        lines.append(f"  {'Stage':<28} {'Calls':>5}  {'Total ms':>10}  {'Avg ms':>9}")
        lines.append("  " + "─" * 56)
        for stage, times in sorted(cls._registry.items()):
            lines.append(
                f"  {stage:<28} {len(times):>5}  {sum(times):>10.2f}  {sum(times) / len(times):>9.2f}"
            )
        lines.append("  " + "─" * 56)
        return "\n".join(lines)

    @classmethod
    def reset(cls) -> None:
        cls._registry.clear()


# ---------------------------------------------------------------------------
# Query Intent Detection
# ---------------------------------------------------------------------------
@dataclass
class QueryIntent:
    """Parsed user intent with type classification and extracted parameters."""
    intent_type: str  # "summary", "comparison", "quote", "theme", "emotion", "general"
    raw_query: str
    search_query: str  # cleaned/optimised query for embedding search
    target_themes: list[str] = field(default_factory=list)
    target_paragraphs: list[int] = field(default_factory=list)
    target_entities: list[str] = field(default_factory=list)
    compare_items: list[str] = field(default_factory=list)


class QueryIntentDetector:
    """
    Detect the type of question the user is asking.

    Supported intent types:
        summary     — "Summarize...", "What are the main...", "Give me an overview..."
        comparison  — "Compare X with Y", "How does X differ from Y..."
        quote       — "Quote about...", "Find a passage...", "Give me a line..."
        theme       — "What themes...", "loneliness in...", queries matching theme dict
        emotion     — "How does X feel...", "What emotions...", "sentiment of..."
        general     — Everything else (standard semantic search)
    """

    # Regex patterns for intent classification
    SUMMARY_PATTERNS = [
        r"\bsummar(y|ize|ise)\b",
        r"\bmain\s+(theme|idea|point|feeling|emotion|message)",
        r"\boverview\b",
        r"\bbrief(ly)?\b.*\bdescri",
        r"\bwhat\s+(is|are)\s+the\s+(main|key|central|primary)",
        r"\bgist\b",
        r"\btl;?dr\b",
        r"\bin\s+a\s+nutshell\b",
        r"\bwhat\s+happens?\b",
        r"\bwhat\s+took?\s+place\b",
        r"\btell\s+me\s+about\b",
        r"\bdescribe\s+the\s+scene\b",
        r"\bwalk\s+me\s+through\b",
        r"\bwhat\s+goes?\s+on\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bcompar(e|ing|ison)\b",
        r"\bdiffer(s|ence|ent)?\b",
        r"\bsimilar(ity|ities)?\s+(between|to|with)\b",
        r"\bcontrast\b",
        r"\bhow\s+does\s+.+\s+(relate|connect)\s+to\b",
        r"\bvs\.?\b",
        r"\bversus\b",
    ]

    QUOTE_PATTERNS = [
        r"\bquote\b",
        r"\bpassage\b",
        r"\bexact\s+(words?|line|sentence|text)\b",
        r"\bgive\s+me\s+a\s+line\b",
        r"\bfind\s+(a\s+)?sentence\b",
        r"\bcite\b",
        r"\bexcerpt\b",
    ]

    EMOTION_PATTERNS = [
        r"\bfeel(s|ing)?\b",
        r"\bemotion(s|al)?\b",
        r"\bsentiment\b",
        r"\bmood\b",
        r"\bhow\s+does\s+.+\s+feel\b",
        r"\bwhat\s+(does|do)\s+.+\s+feel\b",
        r"\bhappy\b|\bsad\b|\bangry\b|\bafraid\b|\bjoyful\b",
    ]

    THEME_PATTERNS = [
        r"\btheme(s)?\b",
        r"\bmotif(s)?\b",
        r"\bsymbol(s|ism|ic)?\b",
        r"\brecurring\b",
        r"\bleitmotif\b",
    ]

    PARAGRAPH_PATTERN = re.compile(
        r"(?:paragraph|para|chapter|section)\s*(\d+)", re.IGNORECASE
    )

    def detect(self, raw_query: str) -> QueryIntent:
        """Classify the user's query and extract structured parameters."""
        query_lower = raw_query.lower().strip()

        # Extract paragraph references
        target_paragraphs = [
            int(m) for m in self.PARAGRAPH_PATTERN.findall(raw_query)
        ]

        # Detect themes mentioned in the query
        target_themes = self._detect_themes_in_query(query_lower)

        # Detect comparison entities
        compare_items = self._extract_comparison_items(query_lower)

        # Classify intent — order matters: more specific intents checked first.
        # IMPORTANT: theme intent must NOT fire just because theme keywords appear
        # in the query — e.g. "what happens on the night they meet" contains
        # "night" (→ nature theme) but is clearly a summary/narrative question.
        intent_type = "general"
        if self._matches(query_lower, self.COMPARISON_PATTERNS) and compare_items:
            intent_type = "comparison"
        elif self._matches(query_lower, self.SUMMARY_PATTERNS):
            intent_type = "summary"
        elif self._matches(query_lower, self.QUOTE_PATTERNS):
            intent_type = "quote"
        elif self._matches(query_lower, self.EMOTION_PATTERNS):
            intent_type = "emotion"
        elif (
            self._matches(query_lower, self.THEME_PATTERNS)
            or (target_themes and not self._matches(query_lower, self.SUMMARY_PATTERNS))
        ):
            intent_type = "theme"

        # Build optimised search query (strip command-like words)
        search_query = self._build_search_query(raw_query, intent_type)

        return QueryIntent(
            intent_type=intent_type,
            raw_query=raw_query,
            search_query=search_query,
            target_themes=target_themes,
            target_paragraphs=target_paragraphs,
            compare_items=compare_items,
        )

    @staticmethod
    def _matches(text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _detect_themes_in_query(query_lower: str) -> list[str]:
        """Check if the query mentions any known theme keywords."""
        found_themes: set[str] = set()
        words = re.findall(r"\b\w+\b", query_lower)
        for w in words:
            if w in _WORD_TO_THEMES:
                found_themes.update(_WORD_TO_THEMES[w])
        # Also check theme names directly
        for theme_name in THEME_DICTIONARY:
            if theme_name in query_lower:
                found_themes.add(theme_name)
        return sorted(found_themes)

    @staticmethod
    def _extract_comparison_items(query_lower: str) -> list[str]:
        """Extract items being compared (e.g., 'Compare X with Y')."""
        patterns = [
            r"compare\s+(.+?)\s+(?:with|to|and|vs\.?)\s+(.+)",
            r"difference\s+between\s+(.+?)\s+and\s+(.+)",
            r"(.+?)\s+vs\.?\s+(.+)",
            r"how\s+does\s+(.+?)\s+(?:relate|connect)\s+to\s+(.+)",
            r"contrast\s+(.+?)\s+(?:with|and)\s+(.+)",
        ]
        for pat in patterns:
            m = re.search(pat, query_lower)
            if m:
                items = [g.strip().rstrip("?.!") for g in m.groups() if g.strip()]
                return items
        return []

    @staticmethod
    def _build_search_query(raw_query: str, intent_type: str) -> str:
        """Remove meta-words that don't help embedding similarity."""
        noise_words = [
            r"\bsummarize\b", r"\bsummary\b", r"\boverview\b",
            r"\bgive\s+me\b", r"\bfind\b", r"\bshow\s+me\b",
            r"\blist\b", r"\btell\s+me\b", r"\bwhat\s+are\b",
            r"\bcan\s+you\b", r"\bplease\b", r"\bthe\s+main\b",
        ]
        cleaned = raw_query
        for nw in noise_words:
            cleaned = re.sub(nw, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else raw_query


# ---------------------------------------------------------------------------
# Sentiment & Theme Analyser
# ---------------------------------------------------------------------------
class SentimentThemeAnalyser:
    """
    Analyse sentences for sentiment polarity and thematic content.

    Sentiment engine priority:
      1. distilbert-base-uncased-finetuned-sst-2-english (transformer, context-aware)
         — understands literary prose, negation, metaphor far better than VADER
      2. VADER  (keyword, fast fallback)
      3. TextBlob (keyword, fallback)
      4. Built-in keyword list expanded for 19th century literary language
    Theme detection uses the global THEME_DICTIONARY via keyword matching.
    """

    # Transformer model — small, fast, runs on CPU, good on literary text
    _TRANSFORMER_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self) -> None:
        self._transformer = None
        self._vader = None

        if TRANSFORMERS_AVAILABLE:
            try:
                log.info("Loading sentiment model: %s", self._TRANSFORMER_MODEL)
                self._transformer = hf_pipeline(
                    "sentiment-analysis",
                    model=self._TRANSFORMER_MODEL,
                    truncation=True,
                    max_length=128,   # literary sentences rarely exceed this
                )
                log.info("Sentiment engine: transformer (%s)", self._TRANSFORMER_MODEL)
            except Exception as e:
                log.warning("Transformer sentiment failed to load (%s) — falling back.", e)
                self._transformer = None

        if self._transformer is None:
            if VADER_AVAILABLE:
                self._vader = SentimentIntensityAnalyzer()
                log.info("Sentiment engine: VADER")
            elif TEXTBLOB_AVAILABLE:
                log.info("Sentiment engine: TextBlob")
            else:
                log.warning("Sentiment engine: built-in keyword fallback")

    def analyse_sentiment(self, text: str) -> dict[str, Any]:
        """
        Return sentiment analysis for a single text string.

        Returns dict with:
            polarity    : float in [-1, 1]
            label       : str emotion label
            confidence  : float 0-1
        """
        polarity = self._score(text)
        label = self._polarity_to_label(polarity)
        return {
            "polarity": round(polarity, 4),
            "label": label,
            "confidence": round(abs(polarity), 4),
        }

    def _score(self, text: str) -> float:
        """Compute polarity score using the best available engine."""
        if self._transformer is not None:
            try:
                result = self._transformer(text[:512])[0]
                # HuggingFace SST-2 returns POSITIVE/NEGATIVE + score in [0,1]
                # Map to [-1, 1]: POSITIVE → +score, NEGATIVE → -score
                raw = float(result["score"])
                return raw if result["label"] == "POSITIVE" else -raw
            except Exception:
                pass  # fall through to next engine

        if self._vader is not None:
            return float(self._vader.polarity_scores(text)["compound"])

        if TEXTBLOB_AVAILABLE:
            return float(TextBlob(text).sentiment.polarity)

        return self._basic_sentiment(text)

    def batch_analyse(
        self, sentences: list[str]
    ) -> tuple[list[dict], list[list[dict]]]:
        """
        Analyse sentiment and themes for a batch of sentences.
        Uses transformer batch inference when available for speed.
        """
        if self._transformer is not None and sentences:
            try:
                # Batch inference — much faster than one-by-one on transformer
                truncated = [s[:512] for s in sentences]
                results = self._transformer(truncated, batch_size=16)
                sentiments = []
                for res in results:
                    raw = float(res["score"])
                    polarity = raw if res["label"] == "POSITIVE" else -raw
                    label = self._polarity_to_label(polarity)
                    sentiments.append({
                        "polarity": round(polarity, 4),
                        "label": label,
                        "confidence": round(abs(polarity), 4),
                    })
            except Exception:
                sentiments = [self.analyse_sentiment(s) for s in sentences]
        else:
            sentiments = [self.analyse_sentiment(s) for s in sentences]

        themes = [self.analyse_themes(s) for s in sentences]
        return sentiments, themes

    def analyse_themes(self, text: str) -> list[dict[str, Any]]:
        """
        Detect themes present in a text using keyword matching.

        Returns list of dicts with:
            theme       : str theme name
            matches     : list of matched keywords
            strength    : int number of keyword hits
        """
        words = set(re.findall(r"\b\w+\b", text.lower()))
        theme_hits: dict[str, list[str]] = defaultdict(list)

        for word in words:
            if word in _WORD_TO_THEMES:
                for theme in _WORD_TO_THEMES[word]:
                    theme_hits[theme].append(word)

        results = []
        for theme, matches in sorted(theme_hits.items(), key=lambda x: -len(x[1])):
            results.append({
                "theme": theme,
                "matches": sorted(set(matches)),
                "strength": len(matches),
            })
        return results

    @staticmethod
    def _polarity_to_label(polarity: float) -> str:
        for label, (lo, hi) in EMOTION_LABELS.items():
            if lo <= polarity < hi:
                return label.replace("_", " ")
        return "very positive" if polarity >= 0.6 else "very negative"

    @staticmethod
    def _basic_sentiment(text: str) -> float:
        """
        Fallback keyword-based sentiment expanded for 19th century literary prose.
        Covers Dostoevsky-era emotional vocabulary that VADER misses entirely.
        """
        positive = {
            # Core positive
            "happy", "joy", "love", "hope", "bright", "beautiful", "wonderful",
            "good", "great", "warm", "light", "smile", "laugh", "delight",
            # Literary / romantic
            "tender", "bliss", "rapture", "enchanted", "dear", "sweet",
            "grateful", "glad", "radiant", "gentle", "serene", "fondness",
            "beloved", "cherish", "adore", "devotion", "passionate", "ecstasy",
            "heavenly", "pure", "innocent", "kind", "gracious", "pleased",
            "content", "peaceful", "blessed", "grateful", "thankful",
        }
        negative = {
            # Core negative
            "sad", "lonely", "dark", "pain", "suffer", "grief", "despair",
            "misery", "tears", "cry", "gloomy", "bleak", "cold", "fear",
            # Literary / Dostoevsky-era
            "anguish", "torment", "bitter", "wretched", "grudge", "wound",
            "killing", "torture", "reproach", "forsaken", "hopeless", "desolate",
            "lament", "sorrow", "weep", "dreary", "forlorn", "melancholy",
            "ache", "longing", "remorse", "shame", "guilt", "abandoned",
            "lost", "broken", "hollow", "empty", "silence", "pale", "cold",
        }
        words = set(re.findall(r"\b\w+\b", text.lower()))
        pos_count = len(words & positive)
        neg_count = len(words & negative)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def aggregate_sentiment(self, sentiments: list[dict]) -> dict[str, Any]:
        """Compute aggregate sentiment statistics over multiple sentences."""
        if not sentiments:
            return {"avg_polarity": 0.0, "label": "neutral", "distribution": {}}

        polarities = [s["polarity"] for s in sentiments]
        avg = sum(polarities) / len(polarities)
        label = self._polarity_to_label(avg)

        distribution: Counter = Counter()
        for s in sentiments:
            distribution[s["label"]] += 1

        return {
            "avg_polarity": round(avg, 4),
            "label": label,
            "min_polarity": round(min(polarities), 4),
            "max_polarity": round(max(polarities), 4),
            "distribution": dict(distribution.most_common()),
        }

    @staticmethod
    def aggregate_themes(theme_lists: list[list[dict]]) -> list[dict]:
        """Aggregate themes across multiple sentences."""
        theme_counter: Counter = Counter()
        theme_words: dict[str, set[str]] = defaultdict(set)

        for themes in theme_lists:
            for t in themes:
                theme_counter[t["theme"]] += t["strength"]
                theme_words[t["theme"]].update(t["matches"])

        results = []
        for theme, count in theme_counter.most_common():
            results.append({
                "theme": theme,
                "total_strength": count,
                "keywords": sorted(theme_words[theme]),
            })
        return results


# ---------------------------------------------------------------------------
# Smart Answer Generator
# ---------------------------------------------------------------------------
class AnswerGenerator:
    """
    Transforms raw retrieval results into coherent, readable answers.

    Capabilities:
        - Deduplicate near-identical sentences
        - Generate glue text between sentences for natural flow
        - Produce structured sections (Summary, Quotes, Analysis)
        - Handle different intent types with tailored formatting
    """

    # Similarity threshold for deduplication
    DEDUP_THRESHOLD = 0.92

    def __init__(
        self,
        analyser: SentimentThemeAnalyser,
        index: "SentenceIndex",
    ) -> None:
        self.analyser = analyser
        self.index = index

    def generate(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        norm_embeddings: np.ndarray,
    ) -> dict[str, Any]:
        """
        Generate a structured answer based on intent type and results.

        Returns a dict with sections:
            summary_paragraph : str
            top_quotes        : list[str]
            analysis          : dict (sentiment + themes)
            cross_references  : list[str]
            raw_results       : list[QueryResult]
        """
        if not results:
            return {
                "summary_paragraph": "No relevant sentences found for your query.",
                "top_quotes": [],
                "analysis": {},
                "cross_references": [],
                "raw_results": [],
            }

        # Deduplicate
        deduped = self._deduplicate(results, norm_embeddings)

        # Analyse sentiment and themes on result sentences
        sentences = [r.sentence for r in deduped]
        sentiments, themes = self.analyser.batch_analyse(sentences)

        # Generate sections based on intent
        handler = {
            "summary": self._handle_summary,
            "comparison": self._handle_comparison,
            "quote": self._handle_quote,
            "theme": self._handle_theme,
            "emotion": self._handle_emotion,
            "general": self._handle_general,
        }

        handler_fn = handler.get(intent.intent_type, self._handle_general)
        answer = handler_fn(intent, deduped, sentiments, themes)

        # Add cross-references for all intents
        answer["cross_references"] = self._extract_cross_references(deduped)
        answer["raw_results"] = deduped

        return answer

    def _deduplicate(
        self,
        results: list["QueryResult"],
        norm_embeddings: np.ndarray,
    ) -> list["QueryResult"]:
        """Remove near-duplicate sentences from results."""
        if len(results) <= 1:
            return results

        kept: list["QueryResult"] = [results[0]]
        kept_indices = [results[0].sentence_id]

        for r in results[1:]:
            is_dup = False
            for ki in kept_indices:
                if ki < norm_embeddings.shape[0] and r.sentence_id < norm_embeddings.shape[0]:
                    sim = float(
                        norm_embeddings[ki] @ norm_embeddings[r.sentence_id]
                    )
                    if sim >= self.DEDUP_THRESHOLD:
                        is_dup = True
                        break
            if not is_dup:
                kept.append(r)
                kept_indices.append(r.sentence_id)

        if len(kept) < len(results):
            log.debug("Deduplicated: %d → %d results", len(results), len(kept))
        return kept

    def _handle_summary(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        sentiments: list[dict],
        themes: list[list[dict]],
    ) -> dict[str, Any]:
        """Generate a summary-style answer."""
        # Enforce cluster diversity: max 2 results per cluster so a broad
        # summary question doesn't return 7 sentences from the same scene.
        seen_clusters: dict[int, int] = {}
        diverse = []
        for r in results:
            count = seen_clusters.get(r.cluster_id, 0)
            if count < 2:
                diverse.append(r)
                seen_clusters[r.cluster_id] = count + 1

        # Order diverse results by position in the book for narrative flow
        ordered = sorted(diverse[:7], key=lambda r: r.sentence_id)

        summary_parts = []
        for i, r in enumerate(ordered):
            text = r.sentence.strip()
            if i > 0:
                glue = self._glue_text(ordered[i - 1].sentence, text)
                if glue:  # only append if non-empty — avoids double spaces
                    summary_parts.append(glue)
            summary_parts.append(text)

        # Filter empty strings before joining
        summary_paragraph = " ".join(p for p in summary_parts if p)

        # Top quotes (most similar, not reordered)
        top_quotes = [f'"{r.sentence}"' for r in results[:3]]

        # Analysis
        agg_sentiment = self.analyser.aggregate_sentiment(sentiments)
        agg_themes = self.analyser.aggregate_themes(themes)

        return {
            "summary_paragraph": summary_paragraph,
            "top_quotes": top_quotes,
            "analysis": {
                "sentiment": agg_sentiment,
                "themes": agg_themes,
                "intent": "summary",
            },
        }

    def _handle_comparison(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        sentiments: list[dict],
        themes: list[list[dict]],
    ) -> dict[str, Any]:
        """Generate a comparison-style answer."""
        items = intent.compare_items
        if len(items) < 2:
            items = ["item A", "item B"]

        # Split results by which comparison item they're most related to
        groups: dict[str, list["QueryResult"]] = {item: [] for item in items}
        for r in results:
            lower_sent = r.sentence.lower()
            best_item = items[0]
            best_count = 0
            for item in items:
                count = lower_sent.count(item.lower())
                if count > best_count:
                    best_count = count
                    best_item = item
            groups[best_item].append(r)

        # Build comparison paragraph
        parts = []
        for item in items:
            group = groups.get(item, [])
            if group:
                sentences_text = " ".join(r.sentence for r in group[:3])
                parts.append(f"Regarding '{item}': {sentences_text}")
            else:
                parts.append(f"Regarding '{item}': No directly relevant passages found.")

        # Find bridging sentences
        bridges = self._find_bridges_between_groups(groups, items)
        if bridges:
            parts.append(
                "Connection: " + " ".join(b.sentence for b in bridges[:2])
            )

        return {
            "summary_paragraph": " ".join(parts),
            "top_quotes": [f'"{r.sentence}"' for r in results[:3]],
            "analysis": {
                "comparison_groups": {
                    item: [r.sentence for r in group[:3]]
                    for item, group in groups.items()
                },
                "sentiment": self.analyser.aggregate_sentiment(sentiments),
                "themes": self.analyser.aggregate_themes(themes),
                "intent": "comparison",
            },
        }

    def _handle_quote(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        sentiments: list[dict],
        themes: list[list[dict]],
    ) -> dict[str, Any]:
        """Return direct quotes with minimal processing."""
        quotes = []
        for r, sent in zip(results[:5], sentiments[:5]):
            quotes.append({
                "text": r.sentence,
                "paragraph_id": r.paragraph_id,
                "similarity": r.similarity,
                "sentiment": sent["label"],
            })

        summary = "Here are the most relevant passages:\n" + "\n".join(
            f'  • "{q["text"]}"  (paragraph {q["paragraph_id"]}, {q["sentiment"]})'
            for q in quotes
        )

        return {
            "summary_paragraph": summary,
            "top_quotes": [f'"{q["text"]}"' for q in quotes],
            "analysis": {
                "quotes_detail": quotes,
                "intent": "quote",
            },
        }

    def _handle_theme(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        sentiments: list[dict],
        themes: list[list[dict]],
    ) -> dict[str, Any]:
        """Generate a theme-focused answer."""
        target = intent.target_themes

        # Filter results that match the target themes
        themed_results = []
        for r, t_list in zip(results, themes):
            matched = [t for t in t_list if t["theme"] in target] if target else t_list
            if matched:
                themed_results.append((r, matched))

        if themed_results:
            ordered = sorted(themed_results, key=lambda x: -sum(t["strength"] for t in x[1]))
            summary_parts = []
            for r, matched_themes in ordered[:5]:
                theme_names = ", ".join(t["theme"] for t in matched_themes)
                summary_parts.append(f'[{theme_names}] "{r.sentence}"')
            summary = (
                f"Thematic analysis for {', '.join(target) if target else 'detected themes'}:\n"
                + "\n".join(f"  • {p}" for p in summary_parts)
            )
        else:
            summary = "No sentences strongly matching the requested themes were found."

        agg_themes = self.analyser.aggregate_themes(themes)

        return {
            "summary_paragraph": summary,
            "top_quotes": [f'"{r.sentence}"' for r in results[:3]],
            "analysis": {
                "target_themes": target,
                "detected_themes": agg_themes,
                "sentiment": self.analyser.aggregate_sentiment(sentiments),
                "intent": "theme",
            },
        }

    def _handle_emotion(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        sentiments: list[dict],
        themes: list[list[dict]],
    ) -> dict[str, Any]:
        """Generate an emotion-focused answer."""
        agg = self.analyser.aggregate_sentiment(sentiments)

        # Group sentences by sentiment label
        emotion_groups: dict[str, list[str]] = defaultdict(list)
        for r, s in zip(results, sentiments):
            emotion_groups[s["label"]].append(r.sentence)

        # Build narrative
        parts = [
            f"The overall emotional tone is {agg['label']} "
            f"(average polarity: {agg['avg_polarity']:.2f})."
        ]

        for emotion, sentences_list in emotion_groups.items():
            if sentences_list:
                example = sentences_list[0]
                trunc = example[:120] + "…" if len(example) > 120 else example
                parts.append(
                    f'{emotion.capitalize()} ({len(sentences_list)} passages): "{trunc}"'
                )

        return {
            "summary_paragraph": " ".join(parts[:1]) + "\n" + "\n".join(
                f"  • {p}" for p in parts[1:]
            ),
            "top_quotes": [f'"{r.sentence}"' for r in results[:3]],
            "analysis": {
                "sentiment": agg,
                "emotion_groups": {
                    k: v[:2] for k, v in emotion_groups.items()
                },
                "themes": self.analyser.aggregate_themes(themes),
                "intent": "emotion",
            },
        }

    def _handle_general(
        self,
        intent: QueryIntent,
        results: list["QueryResult"],
        sentiments: list[dict],
        themes: list[list[dict]],
    ) -> dict[str, Any]:
        """General-purpose answer with all sections."""
        ordered = sorted(results[:5], key=lambda r: r.sentence_id)

        summary_parts = []
        for i, r in enumerate(ordered):
            text = r.sentence.strip()
            if i > 0:
                glue = self._glue_text(ordered[i - 1].sentence, text)
                if glue:
                    summary_parts.append(glue)
            summary_parts.append(text)

        return {
            "summary_paragraph": " ".join(p for p in summary_parts if p),
            "top_quotes": [f'"{r.sentence}"' for r in results[:3]],
            "analysis": {
                "sentiment": self.analyser.aggregate_sentiment(sentiments),
                "themes": self.analyser.aggregate_themes(themes),
                "intent": "general",
            },
        }

    @staticmethod
    def _glue_text(prev_sentence: str, next_sentence: str) -> str:
        """Generate simple transition text between two sentences."""
        transitions = [
            "Furthermore,", "Additionally,", "Moreover,", "In a related vein,",
            "Building on this,", "Along these lines,", "Similarly,",
        ]
        # Use a deterministic but varied selection based on content
        idx = (len(prev_sentence) + len(next_sentence)) % len(transitions)

        # Add glue only between two complete, separate sentences (ends with
        # punctuation AND next starts with a capital) — not mid-sentence fragments.
        if prev_sentence.rstrip()[-1:] in ".!?" and next_sentence[0:1].isupper():
            return transitions[idx]
        return ""  # sentences are fragments — let them join directly

    def _extract_cross_references(
        self, results: list["QueryResult"]
    ) -> list[str]:
        """Identify repeated ideas and cross-cluster connections."""
        cross_refs = []

        # Find sentences that bridge clusters
        clusters_seen: dict[int, list["QueryResult"]] = defaultdict(list)
        for r in results:
            clusters_seen[r.cluster_id].append(r)

        if len(clusters_seen) > 1:
            cluster_ids = list(clusters_seen.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    ci, cj = cluster_ids[i], cluster_ids[j]
                    ri = clusters_seen[ci][0]
                    rj = clusters_seen[cj][0]
                    cross_refs.append(
                        f"Connection between cluster {ci} and {cj}: "
                        f'"{ri.sentence[:60]}…" relates to "{rj.sentence[:60]}…"'
                    )

        # Find bridge sentences from graph neighbors
        for r in results[:3]:
            for nb in r.cross_cluster_neighbors[:1]:
                nb_text = self.index.get_sentence(nb["sentence_id"])
                nb_cid = self.index.graph_index.get(
                    nb["sentence_id"], {}
                ).get("cluster_id", "?")
                cross_refs.append(
                    f"Bridge from cluster {r.cluster_id} → {nb_cid}: "
                    f'"{nb_text[:80]}…"'
                )

        return cross_refs[:5]

    def _find_bridges_between_groups(
        self,
        groups: dict[str, list["QueryResult"]],
        items: list[str],
    ) -> list["QueryResult"]:
        """Find sentences that connect two comparison groups via graph neighbors."""
        if len(items) < 2:
            return []

        ids_a = {r.sentence_id for r in groups.get(items[0], [])}
        ids_b = {r.sentence_id for r in groups.get(items[1], [])}

        bridges = []
        for r in groups.get(items[0], []):
            for nb in r.neighbors:
                if nb["sentence_id"] in ids_b:
                    bridges.append(r)
                    break

        return bridges


# ---------------------------------------------------------------------------
# QueryResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class QueryResult:
    sentence_id: int
    sentence: str
    paragraph_id: Optional[int]
    cluster_id: int
    similarity: float
    centrality: float
    rerank_score: Optional[float] = None
    hybrid_score: Optional[float] = None
    neighbors: list[dict] = field(default_factory=list)
    cross_cluster_neighbors: list[dict] = field(default_factory=list)
    sentiment: Optional[dict] = None
    themes: Optional[list[dict]] = None

    def display_score(self) -> float:
        if self.hybrid_score is not None:
            return self.hybrid_score
        if self.rerank_score is not None:
            return self.rerank_score
        return self.similarity


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load_json(path: str, label: str) -> Any:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    n = len(data) if isinstance(data, list) else "—"
    log.info("%-28s %s records", label, n)
    return data


def _load_npy(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embeddings not found: {path}")
    arr = np.load(path).astype(np.float32)
    log.info("%-28s shape=%s dtype=%s", "embeddings", arr.shape, arr.dtype)
    return arr


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------
class FAISSIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        nlist: int = FAISS_NLIST,
        nprobe: int = FAISS_NPROBE,
        index_path: str = FAISS_INDEX_PATH,
        persist: bool = False,
        use_gpu: bool = False,
    ) -> None:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss not installed.")

        self.d = embeddings.shape[1]
        self.nprobe = nprobe
        self.index_path = index_path
        normed = normalize(embeddings, norm="l2").astype(np.float32)

        if persist and os.path.isfile(index_path):
            with Timer("faiss load from disk", logging.INFO):
                cpu_index = faiss.read_index(index_path)
            log.info("FAISS index loaded from %s (%d vectors)", index_path, cpu_index.ntotal)
        else:
            with Timer("faiss build/train", logging.INFO):
                cpu_index = self._build(normed, nlist)
            if persist:
                self._save(cpu_index, index_path)

        if hasattr(cpu_index, "nprobe"):
            cpu_index.nprobe = nprobe

        if use_gpu and FAISS_GPU and _FAISS_RES is not None:
            with Timer("faiss cpu→gpu transfer", logging.INFO):
                self._index = faiss.index_cpu_to_gpu(_FAISS_RES, 0, cpu_index)
        else:
            self._index = cpu_index

    @staticmethod
    def _build(normed: np.ndarray, nlist: int) -> "faiss.Index":
        n = normed.shape[0]
        if n < nlist * 4:
            idx = faiss.IndexFlatIP(normed.shape[1])
        else:
            quantiser = faiss.IndexFlatIP(normed.shape[1])
            idx = faiss.IndexIVFFlat(
                quantiser, normed.shape[1], nlist, faiss.METRIC_INNER_PRODUCT
            )
            idx.train(normed)
        idx.add(normed)
        return idx

    @staticmethod
    def _save(index: "faiss.Index", path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cpu_index = index
        try:
            cpu_index = faiss.index_gpu_to_cpu(index)
        except Exception:
            pass
        faiss.write_index(cpu_index, path)

    def search(
        self, query: np.ndarray, k: int, allowed_ids: Optional[set[int]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        fetch_k = max(k * RERANK_FETCH_MULT, k) if allowed_ids else k
        q = normalize(query.reshape(1, -1), norm="l2").astype(np.float32)
        with Timer("faiss search"):
            raw_sims, raw_idxs = self._index.search(q, fetch_k)
        raw_sims, raw_idxs = raw_sims[0], raw_idxs[0]
        if allowed_ids:
            mask = np.array([int(i) in allowed_ids for i in raw_idxs], dtype=bool)
            raw_sims, raw_idxs = raw_sims[mask], raw_idxs[mask]
        return raw_sims[:k], raw_idxs[:k]


# ---------------------------------------------------------------------------
# Cluster centroid pre-filter
# ---------------------------------------------------------------------------
class ClusterCentroidIndex:
    def __init__(
        self,
        norm_embeddings: np.ndarray,
        cluster_members: dict[int, list[int]],
        approx_k: int = DEFAULT_APPROX_K,
    ) -> None:
        self.approx_k = approx_k
        self.cluster_ids_ord: list[int] = []
        centroid_list: list[np.ndarray] = []

        with Timer("centroid index build", logging.INFO):
            for cid, sids in cluster_members.items():
                if cid == -1:
                    continue
                vecs = norm_embeddings[sids]
                centroid = vecs.mean(axis=0, keepdims=True)
                centroid = normalize(centroid, norm="l2")[0]
                self.cluster_ids_ord.append(cid)
                centroid_list.append(centroid)

        if centroid_list:
            self.centroids = np.stack(centroid_list, axis=0).astype(np.float32)
        else:
            self.centroids = np.empty((0, norm_embeddings.shape[1]), dtype=np.float32)

    def candidate_ids(
        self, query: np.ndarray, cluster_members: dict[int, list[int]]
    ) -> Optional[set[int]]:
        if self.approx_k <= 0 or self.centroids.shape[0] == 0:
            return None
        with Timer("centroid pre-filter"):
            sims = self.centroids @ query
            k_eff = min(self.approx_k, len(self.cluster_ids_ord))
            top_pos = np.argpartition(sims, -k_eff)[-k_eff:]
            top_cids = [self.cluster_ids_ord[i] for i in top_pos]
        candidates: set[int] = set()
        for cid in top_cids:
            candidates.update(cluster_members.get(cid, []))
        return candidates


# ---------------------------------------------------------------------------
# Sentence index
# ---------------------------------------------------------------------------
class SentenceIndex:
    def __init__(
        self,
        embeddings: np.ndarray,
        sentence_records: list[dict],
        graph_nodes: list[dict],
        cluster_summary: Optional[list[dict]] = None,
        manifest: Optional[list[dict]] = None,
        build_faiss: bool = False,
        persist_faiss: bool = False,
        use_gpu: bool = False,
        approx_clusters: int = DEFAULT_APPROX_K,
    ) -> None:
        # When Phase 2 chunked sentences, len(embeddings) > len(sentence_records).
        # In that case the manifest has one entry per embedding — compare against it.
        expected_len = len(manifest) if manifest is not None else len(sentence_records)
        if len(embeddings) != expected_len:
            raise ValueError(
                f"Embedding/record mismatch: {len(embeddings)} embeddings vs "
                f"{expected_len} {'manifest' if manifest is not None else 'sentence'} records. "
                "Re-run Phase 2."
            )

        self.embeddings = embeddings
        self.sentence_records = sentence_records
        self.norm_embeddings = normalize(embeddings, norm="l2").astype(np.float32)

        self.graph_index: dict[int, dict] = {
            n["sentence_id"]: n for n in graph_nodes
        }
        self.neighbor_map: dict[int, list[dict]] = {
            n["sentence_id"]: n.get("neighbors", []) for n in graph_nodes
        }

        self.cluster_members: dict[int, list[int]] = {}
        for node in graph_nodes:
            cid = node.get("cluster_id", -1)
            self.cluster_members.setdefault(cid, []).append(node["sentence_id"])

        self.cluster_meta: dict[int, dict] = {}
        if cluster_summary:
            for cs in cluster_summary:
                self.cluster_meta[cs["cluster_id"]] = cs

        self.paragraph_ids: list[int] = sorted(
            {r["paragraph_id"] for r in sentence_records if r.get("paragraph_id") is not None}
        )

        self.centrality: dict[int, float] = {}
        if cluster_summary:
            for cs in cluster_summary:
                for entry in cs.get("top_nodes", []):
                    self.centrality[entry["sentence_id"]] = entry.get("centrality", 0.0)

        self.manifest = manifest

        self.faiss_index: Optional[FAISSIndex] = None
        if build_faiss and FAISS_AVAILABLE:
            self.faiss_index = FAISSIndex(
                self.norm_embeddings,
                persist=persist_faiss,
                use_gpu=use_gpu and FAISS_GPU,
            )

        self.centroid_index = ClusterCentroidIndex(
            self.norm_embeddings, self.cluster_members, approx_k=approx_clusters
        )

        log.info(
            "SentenceIndex ready — %d sentences, %d paragraphs, %d clusters",
            len(sentence_records),
            len(self.paragraph_ids),
            len(self.cluster_members),
        )

    def _exact_search(
        self, query: np.ndarray, k: int, mask: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        with Timer("exact cosine search"):
            sims = self.norm_embeddings @ query
            if mask is not None:
                sims = np.where(mask, sims, -np.inf)
            k_eff = min(k, int((sims > -np.inf).sum()))
            if k_eff == 0:
                return np.array([]), np.array([], dtype=int)
            idxs = np.argpartition(sims, -k_eff)[-k_eff:]
            idxs = idxs[np.argsort(sims[idxs])[::-1]]
        return sims[idxs], idxs

    def _build_mask(self, allowed_ids: Optional[set[int]]) -> Optional[np.ndarray]:
        if allowed_ids is None:
            return None
        mask = np.zeros(len(self.embeddings), dtype=bool)
        for aid in allowed_ids:
            if 0 <= aid < len(mask):
                mask[aid] = True
        return mask

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        cluster_filter: Optional[int] = None,
        paragraph_filter: Optional[int] = None,
        use_faiss: bool = False,
        approx_clusters: int = DEFAULT_APPROX_K,
    ) -> list[QueryResult]:
        q = normalize(query_vec.reshape(1, -1), norm="l2")[0].astype(np.float32)

        allowed_ids: Optional[set[int]] = None
        if cluster_filter is not None:
            allowed_ids = set(self.cluster_members.get(cluster_filter, []))
        elif paragraph_filter is not None:
            allowed_ids = {
                i
                for i, r in enumerate(self.sentence_records)
                if r["paragraph_id"] == paragraph_filter
            }
        elif approx_clusters > 0:
            self.centroid_index.approx_k = approx_clusters
            allowed_ids = self.centroid_index.candidate_ids(q, self.cluster_members)

        if use_faiss and self.faiss_index is not None:
            sims, idxs = self.faiss_index.search(q, top_k, allowed_ids)
        else:
            mask = self._build_mask(allowed_ids)
            sims, idxs = self._exact_search(q, top_k, mask)

        results: list[QueryResult] = []
        for sim, idx in zip(sims, idxs):
            idx = int(idx)
            rec = self.sentence_records[idx]
            gnode = self.graph_index.get(idx, {})
            nbs = self.neighbor_map.get(idx, [])
            cross = [nb for nb in nbs if nb.get("cross_cluster")]
            results.append(
                QueryResult(
                    sentence_id=idx,
                    sentence=rec["sentence"],
                    paragraph_id=rec.get("paragraph_id"),
                    cluster_id=gnode.get("cluster_id", -1),
                    similarity=round(float(sim), 6),
                    centrality=self.centrality.get(idx, 0.0),
                    neighbors=nbs,
                    cross_cluster_neighbors=cross,
                )
            )
        return results

    def get_sentence(self, sid: int) -> str:
        if 0 <= sid < len(self.sentence_records):
            return self.sentence_records[sid]["sentence"]
        return "<unknown>"

    def cluster_ids(self) -> list[int]:
        return sorted(k for k in self.cluster_members if k != -1)


# ---------------------------------------------------------------------------
# Query encoder
# ---------------------------------------------------------------------------
class QueryEncoder:
    def __init__(self, model_name: str = EMBED_MODEL_NAME) -> None:
        log.info("Loading embed model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()
        log.info("Embed model ready dim=%d device=%s", self._dim, self.model.device)

    def encode(self, text: str) -> np.ndarray:
        with Timer("encode single query"):
            vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.astype(np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        with Timer("encode batch queries"):
            vecs = self.model.encode(
                texts, convert_to_numpy=True, show_progress_bar=len(texts) > 10
            )
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Cross-encoder re-ranker
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    def __init__(
        self, model_name: str = RERANK_MODEL_NAME, alpha: float = DEFAULT_ALPHA
    ) -> None:
        if not CROSS_ENCODER_AVAILABLE:
            raise RuntimeError("CrossEncoder not available.")
        log.info("Loading re-rank model: %s alpha=%.2f", model_name, alpha)
        self.model = CrossEncoder(model_name)
        self.alpha = alpha

    @staticmethod
    def _minmax(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-10)

    def rerank(
        self, query: str, results: list[QueryResult], alpha: Optional[float] = None
    ) -> list[QueryResult]:
        if not results:
            return results
        a = alpha if alpha is not None else self.alpha
        pairs = [(query, r.sentence) for r in results]
        with Timer("cross-encoder rerank"):
            raw_scores = self.model.predict(pairs)
        cosine_arr = np.array([r.similarity for r in results], dtype=float)
        rerank_arr = np.array(raw_scores, dtype=float)
        norm_cos = self._minmax(cosine_arr)
        norm_rerank = self._minmax(rerank_arr)
        hybrid_arr = a * norm_cos + (1.0 - a) * norm_rerank
        for r, rs, hs in zip(results, raw_scores, hybrid_arr):
            r.rerank_score = round(float(rs), 6)
            r.hybrid_score = round(float(hs), 6)
        results.sort(key=lambda r: r.display_score(), reverse=True)
        return results


# ---------------------------------------------------------------------------
# Export manager
# ---------------------------------------------------------------------------
class ExportManager:
    """Export query results and generated answers to .txt or .json files."""

    def __init__(self, export_dir: str = EXPORT_DIR) -> None:
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_json(
        self,
        query: str,
        answer: dict[str, Any],
        results: list[QueryResult],
        filename: Optional[str] = None,
    ) -> str:
        """Export results as a structured JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r"[^\w\s]", "", query)[:30].strip().replace(" ", "_")
            filename = f"query_{safe_query}_{timestamp}.json"

        filepath = os.path.join(self.export_dir, filename)

        export_data = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": answer.get("summary_paragraph", ""),
            "top_quotes": answer.get("top_quotes", []),
            "analysis": answer.get("analysis", {}),
            "cross_references": answer.get("cross_references", []),
            "results": [
                {
                    "rank": i + 1,
                    "sentence_id": r.sentence_id,
                    "sentence": r.sentence,
                    "paragraph_id": r.paragraph_id,
                    "cluster_id": r.cluster_id,
                    "similarity": r.similarity,
                    "rerank_score": r.rerank_score,
                    "hybrid_score": r.hybrid_score,
                    "centrality": r.centrality,
                    "sentiment": r.sentiment,
                    "themes": r.themes,
                }
                for i, r in enumerate(results)
            ],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)

        return filepath

    def export_txt(
        self,
        query: str,
        answer: dict[str, Any],
        results: list[QueryResult],
        filename: Optional[str] = None,
    ) -> str:
        """Export results as a human-readable text file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_query = re.sub(r"[^\w\s]", "", query)[:30].strip().replace(" ", "_")
            filename = f"query_{safe_query}_{timestamp}.txt"

        filepath = os.path.join(self.export_dir, filename)

        lines = [
            "=" * 80,
            f"QUERY: {query}",
            f"TIME:  {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 80,
            "",
            "── SUMMARY ──────────────────────────────────────────────",
            "",
            textwrap.fill(
                answer.get("summary_paragraph", ""), width=78, initial_indent="  ",
                subsequent_indent="  "
            ),
            "",
            "── TOP QUOTES ───────────────────────────────────────────",
            "",
        ]

        for i, q in enumerate(answer.get("top_quotes", []), 1):
            lines.append(f"  {i}. {q}")
        lines.append("")

        # Analysis section
        analysis = answer.get("analysis", {})
        if analysis:
            lines.append("── ANALYSIS ─────────────────────────────────────────────")
            lines.append("")

            sentiment = analysis.get("sentiment", {})
            if sentiment:
                lines.append(
                    f"  Sentiment: {sentiment.get('label', 'N/A')} "
                    f"(polarity: {sentiment.get('avg_polarity', 'N/A')})"
                )
                dist = sentiment.get("distribution", {})
                if dist:
                    lines.append(f"  Distribution: {dist}")

            themes = analysis.get("themes", [])
            if themes:
                lines.append("  Themes detected:")
                for t in themes[:5]:
                    lines.append(
                        f"    • {t['theme']} (strength: {t.get('total_strength', t.get('strength', 0))})"
                        f" — keywords: {', '.join(t.get('keywords', t.get('matches', [])))}"
                    )
            lines.append("")

        # Cross-references
        cross_refs = answer.get("cross_references", [])
        if cross_refs:
            lines.append("── CROSS-REFERENCES ─────────────────────────────────────")
            lines.append("")
            for cr in cross_refs:
                lines.append(f"  • {cr}")
            lines.append("")

        # Detailed results
        lines.append("── DETAILED RESULTS ─────────────────────────────────────")
        lines.append("")
        for i, r in enumerate(results, 1):
            score = r.display_score()
            lines.append(
                f"  #{i}  [score={score:.4f}]  para={r.paragraph_id}  "
                f"cluster={r.cluster_id}  sid={r.sentence_id}"
            )
            lines.append(textwrap.fill(r.sentence, width=76, initial_indent="      ",
                                       subsequent_indent="      "))
            if r.sentiment:
                lines.append(
                    f"      sentiment: {r.sentiment.get('label', 'N/A')} "
                    f"({r.sentiment.get('polarity', 0):.3f})"
                )
            if r.themes:
                theme_names = ", ".join(t.get("theme", "") for t in r.themes)
                lines.append(f"      themes: {theme_names}")
            lines.append("")

        lines.append("=" * 80)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def _wrap(text: str, indent: int = 6, width: int = CONSOLE_WIDTH) -> str:
    pad = " " * indent
    return textwrap.fill(text, width=width, initial_indent=pad, subsequent_indent=pad)


def display_smart_answer(
    answer: dict[str, Any],
    results: list[QueryResult],
    index: SentenceIndex,
    intent: QueryIntent,
    show_neighbors: bool = True,
    show_raw: bool = False,
    width: int = CONSOLE_WIDTH,
) -> None:
    """Display a polished, sectioned answer to the console."""

    print()
    print("━" * width)
    intent_label = intent.intent_type.upper()
    print(f"  📋 ANSWER  [{intent_label}]")
    print("━" * width)

    # --- Summary section ---
    print()
    print("  ── Summary ──")
    print()
    summary = answer.get("summary_paragraph", "")
    for line in summary.split("\n"):
        print(_wrap(line.strip(), indent=4, width=width))
    print()

    # --- Top Quotes section ---
    quotes = answer.get("top_quotes", [])
    if quotes:
        print("  ── Top Quotes ──")
        print()
        for i, q in enumerate(quotes, 1):
            print(f"    {i}. {q}")
        print()

    # --- Analysis section ---
    analysis = answer.get("analysis", {})
    if analysis:
        print("  ── Analysis ──")
        print()

        # Sentiment
        sentiment = analysis.get("sentiment", {})
        if sentiment:
            label = sentiment.get("label", "N/A")
            polarity = sentiment.get("avg_polarity", "N/A")
            emoji = _sentiment_emoji(label)
            print(f"    {emoji} Sentiment: {label} (polarity: {polarity})")

            dist = sentiment.get("distribution", {})
            if dist:
                dist_str = ", ".join(f"{k}: {v}" for k, v in dist.items())
                print(f"      Distribution: {dist_str}")

        # Themes
        themes = analysis.get("themes", analysis.get("detected_themes", []))
        if themes:
            print("    🏷️  Themes:")
            for t in themes[:5]:
                strength = t.get("total_strength", t.get("strength", 0))
                keywords = t.get("keywords", t.get("matches", []))
                print(f"      • {t['theme']} (strength: {strength}) — {', '.join(keywords[:6])}")

        # Comparison groups
        comp_groups = analysis.get("comparison_groups", {})
        if comp_groups:
            print("    ⚖️  Comparison Groups:")
            for item, sents in comp_groups.items():
                print(f"      [{item}]:")
                for s in sents[:2]:
                    trunc = s[:80] + "…" if len(s) > 80 else s
                    print(f"        • \"{trunc}\"")

        # Emotion groups
        emotion_groups = analysis.get("emotion_groups", {})
        if emotion_groups:
            print("    🎭 Emotion Breakdown:")
            for emotion, sents in emotion_groups.items():
                print(f"      {emotion}:")
                for s in sents[:1]:
                    trunc = s[:80] + "…" if len(s) > 80 else s
                    print(f"        \"{trunc}\"")

        print()

    # --- Cross-references section ---
    cross_refs = answer.get("cross_references", [])
    if cross_refs:
        print("  ── Cross-References ──")
        print()
        for cr in cross_refs:
            print(_wrap(f"• {cr}", indent=4, width=width))
        print()

    # --- Raw results (optional) ---
    if show_raw and results:
        print("  ── Detailed Results ──")
        print()
        for rank, r in enumerate(results, 1):
            score = r.display_score()
            parts = [f"sim={r.similarity:.4f}"]
            if r.rerank_score is not None:
                parts.append(f"rerank={r.rerank_score:.4f}")
            if r.hybrid_score is not None:
                parts.append(f"hybrid={r.hybrid_score:.4f}")
            scores_str = "  ".join(parts)

            print(
                f"    #{rank:<3} [{scores_str}]  "
                f"para={r.paragraph_id}  sid={r.sentence_id}  "
                f"cluster={r.cluster_id}  centrality={r.centrality:.4f}"
            )
            print(_wrap(r.sentence, indent=8, width=width))

            if r.sentiment:
                em = _sentiment_emoji(r.sentiment.get("label", ""))
                print(
                    f"        {em} {r.sentiment.get('label', 'N/A')} "
                    f"(polarity: {r.sentiment.get('polarity', 0):.3f})"
                )
            if r.themes:
                theme_names = ", ".join(t.get("theme", "") for t in r.themes[:3])
                print(f"        🏷️  {theme_names}")

            if show_neighbors and r.neighbors:
                n_show = min(3, len(r.neighbors))
                print(f"        ── neighbors (top {n_show}/{len(r.neighbors)}):")
                for nb in r.neighbors[:n_show]:
                    txt = index.get_sentence(nb["sentence_id"])
                    trunc = txt[:55] + "…" if len(txt) > 58 else txt
                    cross_tag = " [bridge]" if nb.get("cross_cluster") else ""
                    print(
                        f"           [{nb['sentence_id']}] sim={nb['similarity']:.4f}"
                        f"{cross_tag}  \"{trunc}\""
                    )
            print()

    print("━" * width)


def _sentiment_emoji(label: str) -> str:
    """Map a sentiment label to a display emoji."""
    mapping = {
        "very positive": "😄",
        "positive": "🙂",
        "slightly positive": "🙂",
        "neutral": "😐",
        "slightly negative": "😕",
        "negative": "😞",
        "very negative": "😢",
    }
    return mapping.get(label, "🔹")


def display_clusters(index: SentenceIndex) -> None:
    print(f"\n  {'ID':>4}  {'Size':>6}  Centroid (truncated)")
    print("  " + "─" * (CONSOLE_WIDTH - 2))
    for cid in sorted(index.cluster_members):
        meta = index.cluster_meta.get(cid, {})
        size = len(index.cluster_members[cid])
        centroid = meta.get("centroid_sentence", "<no centroid>")[:70]
        noise = " [noise]" if cid == -1 else ""
        print(f"  {cid:>4}{noise}  {size:>6}  \"{centroid}\"")
    print()


def display_help() -> None:
    print("""
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  COMMANDS
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  QUERIES (just type naturally):
    "What does the protagonist feel about loneliness?"
    "Summarize the main themes of Chapter 2"
    "Compare White Nights with modern relationships"
    "Give me a short emotional summary of the ending"
    "Quote about love and dreams"

  SETTINGS:
    :top <N>                 Set result count
    :cluster <ID>|off        Hard cluster filter
    :para <ID>|off           Hard paragraph filter
    :approx <N>|off          Centroid pre-filter clusters
    :rerank on|off           Toggle re-ranking
    :alpha <0.0–1.0>         Cosine weight in hybrid score
    :backend exact|faiss     Switch search backend
    :neighbors on|off        Toggle neighbor display
    :raw on|off              Toggle detailed raw results

  ANALYSIS:
    :themes                  Show all available themes
    :clusters                List clusters with centroids
    :analyse <sentence_id>   Analyse a specific sentence

  EXPORT:
    :save json               Save last results as JSON
    :save txt                Save last results as text
    :save both               Save both formats

  INFO:
    :timing                  Show cumulative timing report
    :timing reset            Reset timing counters
    :help                    Show this message
    :quit / Ctrl+C           Exit
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


# ---------------------------------------------------------------------------
# Batch query mode
# ---------------------------------------------------------------------------
def run_batch(
    query_file: str,
    index: SentenceIndex,
    encoder: QueryEncoder,
    reranker: Optional[CrossEncoderReranker],
    analyser: SentimentThemeAnalyser,
    answer_gen: AnswerGenerator,
    intent_detector: QueryIntentDetector,
    top_k: int = DEFAULT_TOP_K,
    use_faiss: bool = False,
    alpha: float = DEFAULT_ALPHA,
    approx_k: int = DEFAULT_APPROX_K,
    output_path: str = BATCH_OUTPUT_PATH,
) -> None:
    raw_queries = Path(query_file).read_text(encoding="utf-8").splitlines()
    queries = [q.strip() for q in raw_queries if q.strip()]
    if not queries:
        log.warning("Batch file %s contained no queries.", query_file)
        return

    log.info("Batch mode: %d queries from %s", len(queries), query_file)

    with Timer("batch embed all queries", logging.INFO):
        all_vecs = encoder.encode_batch(queries)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out_fh:
        for qi, (query, qvec) in enumerate(zip(queries, all_vecs)):
            intent = intent_detector.detect(query)

            # Use search_query for embedding if different from raw
            if intent.search_query != query:
                qvec = encoder.encode(intent.search_query)

            fetch_k = top_k * RERANK_FETCH_MULT if reranker else top_k
            results = index.search(
                qvec,
                top_k=fetch_k,
                use_faiss=use_faiss,
                approx_clusters=approx_k,
            )
            if reranker and results:
                results = reranker.rerank(query, results, alpha=alpha)
            results = results[:top_k]

            # Enrich with sentiment/themes
            for r in results:
                r.sentiment = analyser.analyse_sentiment(r.sentence)
                r.themes = analyser.analyse_themes(r.sentence)

            # Generate smart answer
            answer = answer_gen.generate(intent, results, index.norm_embeddings)

            record = {
                "query_index": qi,
                "query": query,
                "intent": intent.intent_type,
                "summary": answer.get("summary_paragraph", ""),
                "top_quotes": answer.get("top_quotes", []),
                "analysis": answer.get("analysis", {}),
                "cross_references": answer.get("cross_references", []),
                "results": [
                    {
                        "rank": rank,
                        "sentence_id": r.sentence_id,
                        "sentence": r.sentence,
                        "paragraph_id": r.paragraph_id,
                        "cluster_id": r.cluster_id,
                        "similarity": r.similarity,
                        "rerank_score": r.rerank_score,
                        "hybrid_score": r.hybrid_score,
                        "centrality": r.centrality,
                        "sentiment": r.sentiment,
                        "themes": r.themes,
                    }
                    for rank, r in enumerate(results, 1)
                ],
            }
            out_fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            log.info(
                "[%d/%d] [%s] %-45s → %d results",
                qi + 1, len(queries), intent.intent_type, query[:45], len(results),
            )

    log.info("Batch results written → %s", output_path)


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------
def interactive_loop(
    index: SentenceIndex,
    encoder: QueryEncoder,
    reranker: Optional[CrossEncoderReranker],
    analyser: SentimentThemeAnalyser,
    answer_gen: AnswerGenerator,
    intent_detector: QueryIntentDetector,
    export_manager: ExportManager,
    initial_top_k: int = DEFAULT_TOP_K,
    initial_cluster: Optional[int] = None,
    initial_para: Optional[int] = None,
    use_faiss: bool = False,
    alpha: float = DEFAULT_ALPHA,
    approx_clusters: int = DEFAULT_APPROX_K,
) -> None:
    top_k = initial_top_k
    cluster_filter = initial_cluster
    paragraph_filter = initial_para
    show_neighbors = True
    show_raw = False
    do_rerank = reranker is not None
    backend_faiss = use_faiss and (index.faiss_index is not None)
    cur_alpha = alpha
    cur_approx = approx_clusters

    # State for last query (for export)
    last_query: Optional[str] = None
    last_answer: Optional[dict] = None
    last_results: Optional[list[QueryResult]] = None
    last_intent: Optional[QueryIntent] = None

    print()
    print("━" * CONSOLE_WIDTH)
    print("  🔍 SEMANTIC QUERY INTERFACE")
    print("  Type a natural question or :help for commands.")
    print()
    print("  Examples:")
    print('    "What does the protagonist feel about loneliness?"')
    print('    "Summarize the main themes"')
    print('    "Compare love with despair"')
    print("━" * CONSOLE_WIDTH)

    while True:
        # Build status line
        filters = []
        if cluster_filter is not None:
            filters.append(f"cluster={cluster_filter}")
        if paragraph_filter is not None:
            filters.append(f"para={paragraph_filter}")
        if cur_approx > 0 and not filters:
            filters.append(f"approx={cur_approx}")
        be_tag = "faiss" if backend_faiss else "exact"
        rr_tag = f" +rerank(α={cur_alpha:.1f})" if do_rerank else ""
        status = (
            f"[top={top_k}  {be_tag}{rr_tag}"
            + (f"  {' '.join(filters)}" if filters else "")
            + "]"
        )

        try:
            user_input = input(f"\n  🔍 {status}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!\n")
            print(Timer.report())
            break

        if not user_input:
            continue

        lo = user_input.lower()

        # --- Commands ---
        if lo in (":quit", ":exit", ":q"):
            print("\n  Goodbye!\n")
            print(Timer.report())
            break

        if lo == ":help":
            display_help()
            continue

        if lo == ":clusters":
            display_clusters(index)
            continue

        if lo == ":themes":
            print("\n  Available themes:")
            for theme, keywords in THEME_DICTIONARY.items():
                print(f"    • {theme}: {', '.join(keywords[:8])}…")
            print()
            continue

        if lo == ":timing":
            print(Timer.report())
            continue

        if lo == ":timing reset":
            Timer.reset()
            print("  Timing counters reset.")
            continue

        if lo.startswith(":top"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                top_k = max(1, int(parts[1]))
                print(f"  → top_k = {top_k}")
            else:
                print("  Usage: :top <N>")
            continue

        if lo.startswith(":alpha"):
            parts = user_input.split()
            if len(parts) == 2:
                try:
                    cur_alpha = max(0.0, min(1.0, float(parts[1])))
                    if reranker:
                        reranker.alpha = cur_alpha
                    print(f"  → alpha = {cur_alpha:.2f}")
                except ValueError:
                    print("  Usage: :alpha <0.0–1.0>")
            continue

        if lo.startswith(":approx"):
            parts = user_input.split()
            if len(parts) == 2:
                if parts[1].lower() == "off":
                    cur_approx = 0
                    print("  → Centroid pre-filter disabled.")
                elif parts[1].isdigit():
                    cur_approx = int(parts[1])
                    print(f"  → Centroid pre-filter: top {cur_approx} clusters")
                else:
                    print("  Usage: :approx <N> | :approx off")
            continue

        if lo.startswith(":cluster"):
            parts = user_input.split()
            if len(parts) == 2:
                if parts[1].lower() == "off":
                    cluster_filter = None
                    print("  → Cluster filter removed.")
                elif parts[1].lstrip("-").isdigit():
                    cid = int(parts[1])
                    if cid in index.cluster_members:
                        cluster_filter = cid
                        paragraph_filter = None
                        print(
                            f"  → Cluster {cid} ({len(index.cluster_members[cid])} sentences)"
                        )
                    else:
                        print(f"  Cluster {cid} not found. Available: {index.cluster_ids()}")
                else:
                    print("  Usage: :cluster <ID> | :cluster off")
            continue

        if lo.startswith(":para"):
            parts = user_input.split()
            if len(parts) == 2:
                if parts[1].lower() == "off":
                    paragraph_filter = None
                    print("  → Paragraph filter removed.")
                elif parts[1].isdigit():
                    pid = int(parts[1])
                    if pid in index.paragraph_ids:
                        paragraph_filter = pid
                        cluster_filter = None
                        print(f"  → Paragraph filter: {pid}")
                    else:
                        print(
                            f"  Paragraph {pid} not found. Available: {index.paragraph_ids}"
                        )
                else:
                    print("  Usage: :para <ID> | :para off")
            continue

        if lo.startswith(":rerank"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                want = parts[1].lower() == "on"
                if want and reranker is None:
                    print("  Re-ranker not available.")
                else:
                    do_rerank = want
                    print(f"  → Re-ranking: {'on' if do_rerank else 'off'}")
            else:
                print("  Usage: :rerank on|off")
            continue

        if lo.startswith(":backend"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].lower() in ("exact", "faiss"):
                want_faiss = parts[1].lower() == "faiss"
                if want_faiss and index.faiss_index is None:
                    print("  FAISS index not built. Restart with --backend faiss.")
                else:
                    backend_faiss = want_faiss
                    print(f"  → Backend: {'faiss' if backend_faiss else 'exact'}")
            else:
                print("  Usage: :backend exact|faiss")
            continue

        if lo.startswith(":neighbors"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                show_neighbors = parts[1].lower() == "on"
                print(f"  → Neighbors: {'on' if show_neighbors else 'off'}")
            else:
                print("  Usage: :neighbors on|off")
            continue

        if lo.startswith(":raw"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].lower() in ("on", "off"):
                show_raw = parts[1].lower() == "on"
                print(f"  → Raw results: {'on' if show_raw else 'off'}")
            else:
                print("  Usage: :raw on|off")
            continue

        if lo.startswith(":analyse") or lo.startswith(":analyze"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1].isdigit():
                sid = int(parts[1])
                sent = index.get_sentence(sid)
                if sent != "<unknown>":
                    s = analyser.analyse_sentiment(sent)
                    t = analyser.analyse_themes(sent)
                    em = _sentiment_emoji(s["label"])
                    print(f"\n  Sentence [{sid}]: \"{sent}\"")
                    print(f"  {em} Sentiment: {s['label']} (polarity: {s['polarity']:.4f})")
                    if t:
                        print("  🏷️  Themes:")
                        for th in t:
                            print(
                                f"    • {th['theme']} (strength: {th['strength']}) "
                                f"— {', '.join(th['matches'])}"
                            )
                    else:
                        print("  No themes detected.")
                    print()
                else:
                    print(f"  Sentence {sid} not found.")
            else:
                print("  Usage: :analyse <sentence_id>")
            continue

        if lo.startswith(":save"):
            if last_query is None or last_answer is None or last_results is None:
                print("  No results to save. Run a query first.")
                continue

            parts = user_input.split()
            fmt = parts[1].lower() if len(parts) >= 2 else "both"

            if fmt in ("json", "both"):
                path = export_manager.export_json(last_query, last_answer, last_results)
                print(f"  ✓ Saved JSON → {path}")
            if fmt in ("txt", "both"):
                path = export_manager.export_txt(last_query, last_answer, last_results)
                print(f"  ✓ Saved TXT  → {path}")
            if fmt not in ("json", "txt", "both"):
                print("  Usage: :save json|txt|both")
            continue

        if lo.startswith(":"):
            print(f"  Unknown command: {user_input}. Type :help for commands.")
            continue

        # --- Semantic query with smart processing ---
        t0 = time.perf_counter()

        # Step 1: Detect intent
        intent = intent_detector.detect(user_input)
        log.info("Intent: %s  themes=%s  compare=%s", intent.intent_type,
                 intent.target_themes, intent.compare_items)

        # Step 2: Encode the optimised search query
        qvec = encoder.encode(intent.search_query)

        # Step 3: Search
        fetch_k = top_k * RERANK_FETCH_MULT if do_rerank else top_k
        # For comparison queries, fetch more to cover both sides
        if intent.intent_type == "comparison":
            fetch_k = max(fetch_k, top_k * 3)

        # Apply paragraph filter from intent if specified
        para_for_search = paragraph_filter
        if intent.target_paragraphs and paragraph_filter is None:
            para_for_search = intent.target_paragraphs[0]

        results = index.search(
            qvec,
            top_k=fetch_k,
            cluster_filter=cluster_filter,
            paragraph_filter=para_for_search,
            use_faiss=backend_faiss,
            approx_clusters=cur_approx,
        )

        # For comparison queries, also search for each comparison item separately
        if intent.intent_type == "comparison" and intent.compare_items:
            for item in intent.compare_items:
                item_vec = encoder.encode(item)
                item_results = index.search(
                    item_vec,
                    top_k=top_k,
                    cluster_filter=cluster_filter,
                    paragraph_filter=para_for_search,
                    use_faiss=backend_faiss,
                    approx_clusters=cur_approx,
                )
                # Merge without duplicates
                existing_ids = {r.sentence_id for r in results}
                for ir in item_results:
                    if ir.sentence_id not in existing_ids:
                        results.append(ir)
                        existing_ids.add(ir.sentence_id)

        # Step 4: Re-rank if enabled
        if do_rerank and reranker and results:
            results = reranker.rerank(user_input, results, alpha=cur_alpha)

        results = results[: max(top_k, 10)]  # keep enough for answer generation

        # Step 5: Enrich results with sentiment and themes
        for r in results:
            r.sentiment = analyser.analyse_sentiment(r.sentence)
            r.themes = analyser.analyse_themes(r.sentence)

        # Step 6: Generate smart answer
        answer = answer_gen.generate(intent, results, index.norm_embeddings)

        # Trim to requested top_k for display
        display_results_list = results[:top_k]

        t1 = time.perf_counter()
        print(f"\n  ⏱️  Total: {(t1 - t0) * 1000:.1f} ms  |  Intent: {intent.intent_type}")

        # Step 7: Display
        display_smart_answer(
            answer,
            display_results_list,
            index,
            intent,
            show_neighbors=show_neighbors,
            show_raw=show_raw,
        )

        # Store for export
        last_query = user_input
        last_answer = answer
        last_results = display_results_list
        last_intent = intent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 5: Enhanced Semantic Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--embeddings", default=EMBEDDINGS_PATH)
    p.add_argument("--sentences", default=SENTENCES_PATH)
    p.add_argument("--graph", default=IDEA_GRAPH_PATH)
    p.add_argument("--clusters", default=CLUSTER_PATH)
    p.add_argument("--manifest", default=MANIFEST_PATH)
    p.add_argument("--model", default=EMBED_MODEL_NAME)
    p.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--cluster", type=int, default=None)
    p.add_argument("--paragraph", type=int, default=None)
    p.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help="Cosine weight in hybrid score [0.0–1.0]",
    )
    p.add_argument(
        "--approx_clusters", type=int, default=DEFAULT_APPROX_K,
        help="Centroid pre-filter: top-N clusters (0=off)",
    )
    p.add_argument("--backend", choices=["exact", "faiss"], default="exact")
    p.add_argument("--persist_index", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--batch_file", type=str, default=None)
    p.add_argument("--batch_output", type=str, default=BATCH_OUTPUT_PATH)
    p.add_argument("--export_dir", type=str, default=EXPORT_DIR)
    p.add_argument(
        "--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    log.info("━" * CONSOLE_WIDTH)
    log.info("PHASE 5 — Loading resources…")
    log.info("━" * CONSOLE_WIDTH)

    embeddings = _load_npy(args.embeddings)
    sentences = _load_json(args.sentences, "sentence metadata")
    graph_nodes = _load_json(args.graph, "idea graph")

    cluster_summary = (
        _load_json(args.clusters, "cluster summary")
        if os.path.isfile(args.clusters)
        else None
    )
    manifest = (
        _load_json(args.manifest, "embeddings manifest")
        if os.path.isfile(args.manifest)
        else None
    )

    build_faiss = (args.backend == "faiss") and FAISS_AVAILABLE

    index = SentenceIndex(
        embeddings,
        sentences,
        graph_nodes,
        cluster_summary=cluster_summary,
        manifest=manifest,
        build_faiss=build_faiss,
        persist_faiss=args.persist_index,
        use_gpu=args.gpu,
        approx_clusters=args.approx_clusters,
    )

    encoder = QueryEncoder(args.model)

    reranker: Optional[CrossEncoderReranker] = None
    if args.rerank:
        if CROSS_ENCODER_AVAILABLE:
            reranker = CrossEncoderReranker(alpha=args.alpha)
        else:
            log.warning("CrossEncoder not available — re-ranking disabled.")

    # Initialise enhanced components
    analyser = SentimentThemeAnalyser()
    answer_gen = AnswerGenerator(analyser, index)
    intent_detector = QueryIntentDetector()
    export_manager = ExportManager(args.export_dir)

    # --- Batch mode ---
    if args.batch_file:
        run_batch(
            args.batch_file,
            index,
            encoder,
            reranker,
            analyser,
            answer_gen,
            intent_detector,
            top_k=args.top_k,
            use_faiss=build_faiss,
            alpha=args.alpha,
            approx_k=args.approx_clusters,
            output_path=args.batch_output,
        )
        log.info(Timer.report())
        sys.exit(0)

    # --- Interactive mode ---
    interactive_loop(
        index,
        encoder,
        reranker,
        analyser,
        answer_gen,
        intent_detector,
        export_manager,
        initial_top_k=args.top_k,
        initial_cluster=args.cluster,
        initial_para=args.paragraph,
        use_faiss=build_faiss,
        alpha=args.alpha,
        approx_clusters=args.approx_clusters,
    )

    sys.exit(0)


if __name__ == "__main__":
    main()