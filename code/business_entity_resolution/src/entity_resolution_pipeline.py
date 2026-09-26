#!/usr/bin/env python3
"""Business entity resolution pipeline for the Amazon ML Challenge 2026."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import re
import subprocess
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from unidecode import unidecode
except ImportError:  # Keep the pipeline usable before optional transliteration is installed.
    unidecode = None

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None


LEGAL_SUFFIXES = tuple(
    sorted(
        (
            ("private", "limited"),
            ("pvt", "limited"),
            ("pvt", "ltd"),
            ("private", "ltd"),
            ("public", "limited"),
            ("public", "ltd"),
            ("limited",),
            ("ltd",),
            ("incorporated",),
            ("inc",),
            ("corporation",),
            ("corp",),
            ("llc",),
            ("llp",),
            ("plc",),
            (" Gmbh ".strip().lower(),),
            ("sarl",),
            ("sas",),
            ("sa",),
            ("company",),
            ("co",),
        ),
        key=len,
        reverse=True,
    )
)

ADDRESS_TOKEN_MAP = {
    "rd": "road",
    "rdg": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "boul": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "blk": "block",
    "ctr": "center",
    "col": "colony",
    "no": "number",
    "kh": "khasra",
    "sec": "sector",
    "dist": "district",
    "opp": "opposite",
    "nr": "near",
    "r": "rue",
    "appt": "apartment",
    "bd": "boulevard",
    "rte": "route",
    "che": "chemin",
    "imp": "impasse",
    "res": "residence",
    "bat": "building",
    "n": "north",
    "s": "south",
    "e": "east",
    "w": "west",
}

BUSINESS_NAME_TOKEN_MAP = {
    "intl": "international",
    "intnl": "international",
    "mfg": "manufacturing",
    "tech": "technology",
    "technol": "technology",
    "indl": "industrial",
    "svc": "services",
    "svcs": "services",
    "hosp": "hospital",
}

COUNTRY_ALIASES = {
    "usa": "us",
    "united states": "us",
    "united states of america": "us",
    "in": "india",
    "republic of india": "india",
    "fr": "france",
    "french republic": "france",
}
COUNTRY_NAME_COMPONENTS = {
    "us", "usa", "united states", "united states of america",
    "india", "republic of india", "france", "french republic",
}

BLOCK_STOPWORDS = {
    "a", "an", "and", "at", "by", "co", "company", "corp", "corporation",
    "for", "from", "global", "group", "inc", "incorporated", "india", "international",
    "limited", "llc", "llp", "ltd", "of", "on", "pvt", "private", "services", "the",
    "to", "trading", "usa", "us", "solutions", "technology", "technologies",
    "street", "road", "avenue", "boulevard", "drive", "lane", "highway", "parkway",
    "suite", "apartment", "building", "floor", "unit", "near", "opposite", "khasra",
}

US_STATE_CODE_TO_NAME = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
US_STATE_NAME_TO_CODE = {name: code for code, name in US_STATE_CODE_TO_NAME.items()}
INDIAN_STATE_NAMES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
    "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh",
    "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab",
    "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal", "delhi", "jammu and kashmir", "ladakh", "puducherry", "chandigarh",
    "andaman and nicobar islands", "dadra and nagar haveli and daman and diu", "lakshadweep",
}
INDIAN_STATE_ABBREVIATIONS = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "ch": "chandigarh", "ct": "chhattisgarh",
    "dl": "delhi", "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim", "tn": "tamil nadu",
    "ts": "telangana", "tr": "tripura", "up": "uttar pradesh", "uk": "uttarakhand",
    "wb": "west bengal", "jk": "jammu and kashmir",
}

FEATURE_NAMES = [
    "name_exact", "name_core_exact", "name_token_jaccard", "name_token_containment",
    "name_char_similarity", "name_jaro_winkler", "name_char_ngram_jaccard", "name_char_ngram_cosine",
    "name_tfidf_cosine", "name_sorted_token_similarity", "name_token_order_similarity",
    "candidate_retrieval_rank_reciprocal", "candidate_retrieval_score",
    "name_prefix_similarity",
    "name_suffix_similarity", "name_variant_similarity", "name_length_ratio", "common_name_tokens",
    "name_rare_token_overlap",
    "address_exact", "address_token_jaccard", "address_token_containment",
    "address_char_similarity", "address_char_ngram_jaccard", "address_char_ngram_cosine",
    "address_tfidf_cosine", "address_length_ratio",
    "common_address_tokens", "address_rare_token_overlap", "address_landmark_overlap",
    "number_token_jaccard", "house_number_match", "house_number_conflict",
    "unit_number_match", "unit_number_conflict", "postal_code_match", "postal_code_conflict",
    "city_token_jaccard", "city_exact", "state_match", "state_conflict",
    "tail_component_similarity", "same_country", "country_conflict", "name_address_consistency",
    "city_name_consistency",
    "name_postal_consistency", "phone_fragment_match", "cross_source_cluster_size",
    "cross_source_support_count", "s2_s3_name_similarity", "s2_s3_address_similarity",
    "s2_s3_cluster_agreement", "cross_source_candidate_support_count",
    "cross_source_candidate_agreement", "cross_source_direct_edge_count",
    "cross_source_max_edge_score", "cross_source_min_edge_score",
    "cross_source_avg_edge_score", "candidate_source_s2", "candidate_source_s3",
    "candidate_retrieval_channel_count",
    "candidate_block_name_phonetic",
    "candidate_block_name_tfidf",
    "contradiction_score",
    "address_layout_country_plausibility", "address_layout_pattern_match",
    "postal_country_pattern_support", "has_name_both", "has_address_both",
]


class BlockIndex(defaultdict):
    """Posting lists plus target-corpus document frequencies for TF-IDF features."""

    def __init__(self) -> None:
        super().__init__(set)
        self.document_count = 0
        self.name_document_frequency: Counter[str] = Counter()
        self.address_document_frequency: Counter[str] = Counter()
        self.key_document_frequency: Counter[Tuple[str, str]] = Counter()
        self.target_graph_features: Dict[str, Dict[str, float]] = {}
        self.target_graph_neighbors: Dict[str, Set[str]] = {}
        self.country_postal_format_counts: Dict[str, Counter[int]] = defaultdict(Counter)
        self.country_address_layout_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.name_ngram_limit = 6
        self.channel_limits = dict(BLOCK_CHANNEL_LIMITS)
        self.retrieval_context_mode = "once"
        self.semantic_vectorizer: TfidfVectorizer | None = None
        self.semantic_matrix: Any | None = None
        self.semantic_entity_ids = np.asarray([], dtype=object)
        self.semantic_top_k = 50
        self.semantic_min_similarity = 0.12


@dataclass(frozen=True)
class RecordFeatureProfile:
    """Reusable, immutable normalization results for one source record."""

    name_raw: str
    name_variants: Tuple[Tuple[str, ...], ...]
    name_tokens: Tuple[str, ...]
    name_core: str
    address_raw: str
    address_tokens: Tuple[str, ...]
    country: str
    city_tokens: Tuple[str, ...]
    state_tokens: frozenset[str]
    number_tokens: frozenset[str]
    unit_numbers: frozenset[str]
    phone_fragments: frozenset[str]
    postal_codes: frozenset[str]
    layout_signature: str
    house_number: str
    landmark_tokens: frozenset[str]


@dataclass
class NeuralReranker:
    """Classical candidate ranker followed by a small MLP over pair features."""

    base_model: Any
    reranker: Any
    top_k: int


@dataclass
class EnsemblePairEstimator:
    """Average positive-class probabilities from independently fit pair models."""

    models: List[Any]

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([0, 1])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsemblePairEstimator":
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.tile(np.asarray([[0.5, 0.5]]), (len(X), 1))
        positive = np.mean([model.predict_proba(X)[:, 1] for model in self.models], axis=0)
        return np.column_stack((1.0 - positive, positive))


@dataclass
class SourceSpecificPairEstimator:
    """Dispatch to S2 or S3 pair models, with a shared fallback per source."""

    shared_model: Any
    source2_model: Any | None = None
    source3_model: Any | None = None
    source_feature_index: int = 0

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(self.shared_model.classes_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SourceSpecificPairEstimator":
        self.shared_model.fit(X, y)
        source_values = X[:, self.source_feature_index] >= 0.5
        for source_mask, attr in ((source_values, "source2_model"),
                                  (~source_values, "source3_model")):
            if np.count_nonzero(source_mask) and len(np.unique(y[source_mask])) == 2:
                source_model = make_base_pair_estimator(self.base_model_type, self.seed)
                source_model.fit(X[source_mask], y[source_mask])
                setattr(self, attr, source_model)
            else:
                setattr(self, attr, None)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        source_values = X[:, self.source_feature_index] >= 0.5
        probabilities = np.empty((len(X), len(self.classes_)), dtype=float)
        for source_mask, model in ((source_values, self.source2_model),
                                   (~source_values, self.source3_model)):
            if np.any(source_mask):
                selected_model = model if model is not None else self.shared_model
                probabilities[source_mask] = selected_model.predict_proba(X[source_mask])
        return probabilities

    base_model_type: str = "lightgbm"
    seed: int = 42


@dataclass
class CandidateDiagnostics:
    entity_count: int = 0
    example_build_seconds: float = 0.0
    total_truth_pairs: int = 0
    found_truth_pairs: int = 0
    candidate_size_histogram: Counter[int] = field(default_factory=Counter)
    block_size_histograms: Dict[str, Counter[int]] = field(default_factory=lambda: defaultdict(Counter))
    block_truth_hits: Counter[str] = field(default_factory=Counter)
    recall_at_k_hits: Counter[int] = field(default_factory=Counter)
    candidates_at_k: Counter[int] = field(default_factory=Counter)


@dataclass
class OOFModelResult:
    model_type: str
    pair_rows: pd.DataFrame
    entity_features: np.ndarray
    entity_ids: List[str]
    fold_by_entity: np.ndarray
    mined_negatives: pd.DataFrame
    holdout_examples: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_source1: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdout_base_probabilities: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    candidate_recall_by_fold: Dict[int, float] = field(default_factory=dict)
    topk_candidate_recall_by_fold: Dict[int, float] = field(default_factory=dict)
    match_recall_by_fold: Dict[int, float] = field(default_factory=dict)
    probability_calibrator: Any | None = None
    crossfit_probabilities: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    threshold: float = 0.5
    score_margin: float | None = None
    cross_validation_f0_5: float = 0.0
    cross_validation_fold_scores: Dict[int, float] = field(default_factory=dict)
    heldout_metrics: Dict[str, float] = field(default_factory=dict)
    model_runtime_seconds: float = 0.0


@dataclass
class EntityDecisionLayer:
    classifier: Any | None = None
    threshold: float = 0.5
    probability_calibrator: Any | None = None
    cardinality_classifier: Any | None = None
    cardinality_probability_calibrator: Any | None = None
    cardinality_thresholds: Dict[int, float] = field(default_factory=dict)
    cardinality_margins: Dict[int, float | None] = field(default_factory=dict)
    cardinality_confidence_thresholds: Dict[int, float] = field(default_factory=dict)
    feature_pair_threshold: float = 0.0


@dataclass(frozen=True)
class NormalizationSettings:
    strip_legal_suffixes: bool = True
    split_dba_aliases: bool = True
    normalize_address_abbreviations: bool = True


NORMALIZATION_SETTINGS = NormalizationSettings()


ENTITY_FEATURE_NAMES = (
    "best_probability", "second_probability", "probability_gap", "top3_probability_mean",
    "probability_std", "log_candidate_count", "number_above_pair_threshold",
    "best_name_score", "best_address_score", "best_country_score",
    "best_name_rank_reciprocal", "best_address_rank_reciprocal",
    "best_candidate_rank_reciprocal", "logit_gap", "top1_top2_ratio", "top1_top3_gap",
    "top1_name_score", "top2_name_score", "top1_address_score", "top2_address_score",
    "top1_postal_match", "top2_postal_match", "top1_house_match", "top2_house_match",
    "exact_name_count", "exact_address_count", "exact_postal_count",
    "name_exact_block_count", "name_pair_block_count", "name_char_block_count",
    "name_phonetic_block_count",
    "name_tfidf_block_count",
    "name_token_block_count", "address_token_block_count", "postal_block_count",
    "house_number_block_count", "city_token_block_count", "s2_candidate_count",
    "s3_candidate_count", "best_s2_probability", "best_s3_probability",
    "best_s2_s3_probability_gap",
    "top1_country_conflict", "top2_country_conflict",
    "top1_postal_conflict", "top2_postal_conflict",
    "top1_house_conflict", "top2_house_conflict",
    "top1_exact_name", "top2_exact_name",
    "top1_cross_source_support", "top2_cross_source_support",
)
BLOCK_TYPE_NAMES = (
    "name_exact", "name_pair", "name_char_ngram", "name_phonetic", "name_tfidf",
    "name_token", "address_token",
    "postal", "house_number", "city_token",
)
BLOCK_RETRIEVAL_WEIGHTS = {
    "name_exact": 10.0, "name_pair": 8.0, "name_char_ngram": 4.0,
    "name_phonetic": 3.0,
    "name_tfidf": 8.0,
    "name_token": 7.0, "address_token": 2.5, "postal": 9.0,
    "house_number": 7.0, "city_token": 1.5,
}
BLOCK_CHANNEL_LIMITS = {
    "name_exact": 100, "name_pair": 150, "name_char_ngram": 100,
    "name_phonetic": 60,
    "name_tfidf": 50,
    "name_token": 200, "address_token": 200, "postal": 100,
    "house_number": 100, "city_token": 50,
}
BLOCK_FREQUENCY_MULTIPLIERS = {
    "name_exact": 4, "name_pair": 2, "name_char_ngram": 1,
    "name_phonetic": 1,
    "name_tfidf": 1,
    "name_token": 1, "address_token": 1, "postal": 4,
    "house_number": 4, "city_token": 1,
}
CANDIDATE_EFFICIENCY_K = (50, 100, 200, 500, 1000)
PHONETIC_FALLBACK_MIN_SYMBOLIC_CANDIDATES = 5


@lru_cache(maxsize=250_000)
def normalize_text(value: object) -> str:
    """Fold accents and (when available) transliterate non-Latin text for matching."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).casefold().strip().replace("&", " and ").replace("/", " ")
    if unidecode is not None:
        text = unidecode(text)
    else:
        text = "".join(
            char for char in unicodedata.normalize("NFKD", text)
            if not unicodedata.combining(char)
        )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_country(value: object) -> str:
    country = normalize_text(value)
    return COUNTRY_ALIASES.get(country, country)


@lru_cache(maxsize=250_000)
def _normalized_token_tuple(text: str) -> Tuple[str, ...]:
    return tuple(text.split()) if text else ()


def token_list(value: object) -> List[str]:
    return list(_normalized_token_tuple(normalize_text(value)))


def strip_legal_suffix(tokens: Sequence[str]) -> List[str]:
    tokens = list(tokens)
    changed = True
    while tokens and changed:
        changed = False
        for suffix in LEGAL_SUFFIXES:
            if len(tokens) >= len(suffix) and tuple(tokens[-len(suffix):]) == suffix:
                del tokens[-len(suffix):]
                changed = True
                break
    return tokens


DBA_MARKERS = (
    ("doing", "business", "as"),
    ("formerly", "known", "as"),
    ("trading", "as"),
    ("trading", "under"),
    ("t", "a"),
    ("f", "k", "a"),
    ("d", "b", "a"),
    ("dba",),
    ("aka",),
)


def name_variants(value: object) -> List[List[str]]:
    tokens = [BUSINESS_NAME_TOKEN_MAP.get(token, token) for token in token_list(value)]
    if NORMALIZATION_SETTINGS.split_dba_aliases:
        for position in range(len(tokens)):
            for marker in DBA_MARKERS:
                if tuple(tokens[position:position + len(marker)]) == marker:
                    primary = tokens[:position]
                    alias = tokens[position + len(marker):]
                    if NORMALIZATION_SETTINGS.strip_legal_suffixes:
                        primary = strip_legal_suffix(primary)
                        alias = strip_legal_suffix(alias)
                    variants = [variant for variant in (primary, alias) if variant]
                    if variants:
                        return variants
    cleaned = (strip_legal_suffix(tokens) if NORMALIZATION_SETTINGS.strip_legal_suffixes
               else tokens)
    return [cleaned] if cleaned else []


def core_name_tokens(value: object) -> List[str]:
    variants = name_variants(value)
    return variants[0] if variants else []


def soundex_code(value: object) -> str:
    """Return a compact phonetic key after normalizing and transliterating text."""
    letters = re.sub(r"[^a-z]", "", normalize_text(value))
    if not letters:
        return ""
    groups = {
        **dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"), "l": "4", **dict.fromkeys("mn", "5"), "r": "6",
    }
    output = [letters[0].upper()]
    previous = groups.get(letters[0], "")
    for letter in letters[1:]:
        if letter in "hw":
            # Canonical Soundex ignores H/W without breaking adjacent classes.
            continue
        code = groups.get(letter, "")
        if code and code != previous:
            output.append(code)
        previous = code
        if len(output) == 4:
            break
    return "".join(output).ljust(4, "0")


def address_tokens(value: object) -> List[str]:
    tokens = token_list(value)
    if not NORMALIZATION_SETTINGS.normalize_address_abbreviations:
        return tokens
    return [ADDRESS_TOKEN_MAP.get(token, token) for token in tokens]


def address_components(value: object, country: object = "") -> List[List[str]]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    country_key = normalize_country(country)
    components: List[List[str]] = []
    for part in re.split(r"[,;\n]+", str(value)):
        tokens = address_tokens(part)
        if not tokens:
            continue
        component_text = " ".join(tokens)
        component_country = normalize_country(component_text)
        if component_country == country_key and country_key:
            continue
        if component_text in COUNTRY_NAME_COMPONENTS:
            continue
        components.append(tokens)
    return components


def _state_match(tokens: Sequence[str], country: str) -> Tuple[str, int] | None:
    words = [token for token in tokens if not re.fullmatch(r"\d{5,6}", token)]
    if country == "us":
        for size in (3, 2, 1):
            if len(words) < size:
                continue
            phrase = " ".join(words[-size:])
            if phrase in US_STATE_NAME_TO_CODE:
                return US_STATE_CODE_TO_NAME[US_STATE_NAME_TO_CODE[phrase]], len(words) - size
            if size == 1 and phrase in US_STATE_CODE_TO_NAME:
                return US_STATE_CODE_TO_NAME[phrase], len(words) - 1
    elif country == "india":
        for size in (4, 3, 2, 1):
            if len(words) >= size:
                phrase = " ".join(words[-size:])
                if phrase in INDIAN_STATE_NAMES:
                    return phrase, len(words) - size
                if size == 1 and phrase in INDIAN_STATE_ABBREVIATIONS:
                    return INDIAN_STATE_ABBREVIATIONS[phrase], len(words) - 1
    return None


def _without_postal_and_numbers(tokens: Sequence[str]) -> List[str]:
    return [token for token in tokens if not re.fullmatch(r"\d{5,6}|\d+[a-z]?", token)]


def _locality_tokens(tokens: Sequence[str]) -> List[str]:
    locality = _without_postal_and_numbers(tokens)
    # In trailing locality components, "St Louis" and "St-Denis" mean Saint,
    # while the same abbreviation in a street component means Street.
    if locality and locality[0] == "street":
        locality[0] = "saint"
    return locality


def address_city_tokens(value: object, country: object = "") -> List[str]:
    """Extract a likely locality from trailing comma-separated address components."""
    components = address_components(value, country)
    if not components:
        return []
    country_key = normalize_country(country)

    if country_key in {"us", "india"}:
        # State and postal code usually share the last component; city is usually
        # the component immediately before it. Also handle "City, ST 12345".
        for index in range(len(components) - 1, max(-1, len(components) - 3), -1):
            state = _state_match(components[index], country_key)
            if state is None:
                continue
            _, state_start = state
            stripped = _without_postal_and_numbers(components[index])
            prefix = stripped[:min(state_start, len(stripped))]
            if prefix:
                return _locality_tokens(prefix)
            if index > 0:
                return _locality_tokens(components[index - 1])
            return []

    # French and many international forms put the postal code beside the city:
    # "75001 Paris" or "Paris, 75001". If the postal component is digits only,
    # use the preceding component. A single unstructured line is not guessed to
    # be a city because it is usually the street address.
    for index in range(len(components) - 1, -1, -1):
        component = components[index]
        postal_positions = [position for position, token in enumerate(component)
                            if re.fullmatch(r"\d{5,6}", token)]
        if postal_positions:
            after_postal = _without_postal_and_numbers(component[postal_positions[-1] + 1:])
            if after_postal:
                return _locality_tokens(after_postal)
            if index > 0:
                return _locality_tokens(components[index - 1])
            return []
    if len(components) >= 2:
        return _locality_tokens(components[-1])
    return []


def address_state_tokens(value: object, country: object = "") -> List[str]:
    country_key = normalize_country(country)
    if country_key not in {"us", "india"}:
        return []
    components = address_components(value, country)
    for component in reversed(components[-3:]):
        state = _state_match(component, country_key)
        if state is not None:
            return token_list(state[0])
    return []


def address_numbers(value: object) -> Set[str]:
    return {token for token in address_tokens(value) if re.fullmatch(r"\d+[a-z]?", token)}


def postal_codes(value: object) -> Set[str]:
    text = normalize_text(value)
    return set(re.findall(r"(?<!\d)\d{5,6}(?!\d)", text))


def address_layout_signature(value: object, country: object = "") -> str:
    """Return a small address-shape key for data-driven country profiles."""
    components = address_components(value, country)
    component_bucket = min(len(components), 4)
    postal_component = next(
        (position for position, component in enumerate(components)
         if re.search(r"(?<!\d)\d{5,6}(?!\d)", " ".join(component))),
        None,
    )
    if postal_component is None:
        postal_position = "none"
    elif postal_component == len(components) - 1:
        postal_position = "last"
    elif postal_component == 0:
        postal_position = "first"
    else:
        postal_position = "middle"
    first_component = components[0] if components else []
    starts_with_number = bool(first_component and re.fullmatch(r"\d+[a-z]?", first_component[0]))
    number_components = [
        position for position, component in enumerate(components)
        if any(re.fullmatch(r"\d+[a-z]?", token) for token in component)
    ]
    if not number_components:
        number_position = "none"
    elif number_components[0] == 0:
        number_position = "first"
    elif number_components[0] == len(components) - 1:
        number_position = "last"
    else:
        number_position = "middle"
    first_component_shape = (
        "number_first" if starts_with_number else
        "number_later" if any(re.fullmatch(r"\d+[a-z]?", token) for token in first_component) else
        "text" if first_component else "empty"
    )
    if postal_component is None:
        postal_order = "none"
    else:
        tokens = components[postal_component]
        postal_token_position = next(
            (position for position, token in enumerate(tokens)
             if re.fullmatch(r"\d{5,6}", token)),
            -1,
        )
        has_text_before = any(re.search(r"[a-z]", token) for token in tokens[:postal_token_position])
        has_text_after = any(re.search(r"[a-z]", token) for token in tokens[postal_token_position + 1:])
        postal_order = "text_before" if has_text_before and not has_text_after else (
            "text_after" if has_text_after and not has_text_before else "mixed"
        )
    return (f"c{component_bucket}|p{postal_position}|o{postal_order}|"
            f"n{int(starts_with_number)}|h{number_position}|f{first_component_shape}")


def smoothed_profile_frequency(counts: Mapping[object, int], key: object) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    categories = max(len(counts) + int(key not in counts), 1)
    return float((counts.get(key, 0) + 1) / (total + categories))


def house_number(value: object) -> str:
    first_component = str(value or "").split(",", 1)[0]
    numbers = re.findall(r"(?<![a-z])\d+[a-z]?(?![a-z])", normalize_text(first_component))
    return numbers[0] if numbers else ""


def unit_numbers(value: object) -> Set[str]:
    tokens = address_tokens(value)
    markers = {"unit", "apartment", "suite", "floor", "number"}
    return {tokens[i + 1] for i, token in enumerate(tokens[:-1])
            if token in markers and re.fullmatch(r"\d+[a-z]?", tokens[i + 1])}


def address_landmark_tokens(value: object) -> Set[str]:
    markers = {"near", "opposite", "behind", "beside", "adjacent", "landmark"}
    landmarks: Set[str] = set()
    for component in address_components(value):
        for position, token in enumerate(component):
            if token not in markers:
                continue
            for following in component[position + 1:position + 5]:
                if re.fullmatch(r"\d+[a-z]?", following):
                    break
                if following not in BLOCK_STOPWORDS:
                    landmarks.add(following)
    return landmarks


@lru_cache(maxsize=50_000)
def _cached_record_feature_profile(
    name: str,
    address: str,
    country: str,
    strip_legal_suffixes: bool,
    split_dba_aliases: bool,
    normalize_address_abbreviations: bool,
) -> RecordFeatureProfile:
    """Parse row-level features once, then reuse them across candidate pairs."""
    # The normalization settings are part of the cache key so normalization
    # ablations cannot accidentally reuse profiles from another configuration.
    if (
        NORMALIZATION_SETTINGS.strip_legal_suffixes != strip_legal_suffixes
        or NORMALIZATION_SETTINGS.split_dba_aliases != split_dba_aliases
        or NORMALIZATION_SETTINGS.normalize_address_abbreviations != normalize_address_abbreviations
    ):
        raise RuntimeError("Normalization settings changed while building a cached record profile.")
    variants = tuple(tuple(variant) for variant in name_variants(name))
    name_tokens = tuple(variants[0]) if variants else ()
    country_key = normalize_country(country)
    return RecordFeatureProfile(
        name_raw=normalize_text(name),
        name_variants=variants,
        name_tokens=name_tokens,
        name_core=" ".join(name_tokens),
        address_raw=normalize_text(address),
        address_tokens=tuple(address_tokens(address)),
        country=country_key,
        city_tokens=tuple(address_city_tokens(address, country_key)),
        state_tokens=frozenset(address_state_tokens(address, country_key)),
        number_tokens=frozenset(address_numbers(address)),
        unit_numbers=frozenset(unit_numbers(address)),
        phone_fragments=frozenset(phone_fragments(address)),
        postal_codes=frozenset(postal_codes(address)),
        layout_signature=address_layout_signature(address, country_key),
        house_number=house_number(address),
        landmark_tokens=frozenset(address_landmark_tokens(address)),
    )


def record_feature_profile(row: Mapping[str, object] | pd.Series) -> RecordFeatureProfile:
    name = _row_value(row, "business_name")
    address = _row_value(row, "business_address")
    country = _row_value(row, "country")
    name_key = "" if name is None or (isinstance(name, float) and np.isnan(name)) else str(name)
    address_key = "" if address is None or (isinstance(address, float) and np.isnan(address)) else str(address)
    country_key = "" if country is None or (isinstance(country, float) and np.isnan(country)) else str(country)
    return _cached_record_feature_profile(
        name_key, address_key, country_key,
        NORMALIZATION_SETTINGS.strip_legal_suffixes,
        NORMALIZATION_SETTINGS.split_dba_aliases,
        NORMALIZATION_SETTINGS.normalize_address_abbreviations,
    )


def phone_fragments(value: object) -> Set[str]:
    raw = str(value or "")
    fragments = set()
    for match in re.findall(r"\+?\d[\d\s()./-]{6,}\d", raw):
        digits = re.sub(r"\D", "", match)
        if 8 <= len(digits) <= 15:
            fragments.add(digits)
    return fragments


def set_jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def containment_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def length_ratio(a: Sequence[object], b: Sequence[object]) -> float:
    if not a or not b:
        return 0.0
    return min(len(a), len(b)) / max(len(a), len(b))


def char_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if Levenshtein is not None:
        return float(Levenshtein.ratio(a, b))
    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())


def token_order_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, list(a), list(b), autojunk=False).ratio())


def char_ngrams(text: str, size: int = 3) -> Set[str]:
    compact = text.replace(" ", "")
    if not compact:
        return set()
    if len(compact) < size:
        return {compact}
    return {compact[i:i + size] for i in range(len(compact) - size + 1)}


def char_ngram_cosine(a: str, b: str, size: int = 3) -> float:
    left, right = char_ngrams(a, size), char_ngrams(b, size)
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def jaro_winkler_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if Levenshtein is not None and hasattr(Levenshtein, "jaro_winkler"):
        return float(Levenshtein.jaro_winkler(a, b))
    # SequenceMatcher is the dependency-free fallback; it remains a useful typo signal.
    return char_similarity(a, b)


def tfidf_cosine(a: Sequence[str], b: Sequence[str], document_frequency: Mapping[str, int],
                 document_count: int) -> float:
    left_counts, right_counts = Counter(a), Counter(b)
    if not left_counts or not right_counts:
        return 0.0

    def vector(counts: Counter[str]) -> Dict[str, float]:
        weighted: Dict[str, float] = {}
        for token, term_count in counts.items():
            df = document_frequency.get(token, 0)
            idf = math.log((1.0 + document_count) / (1.0 + df)) + 1.0
            weighted[token] = (1.0 + math.log(term_count)) * idf
        return weighted

    left_vector, right_vector = vector(left_counts), vector(right_counts)
    numerator = sum(left_vector[token] * right_vector[token]
                    for token in left_vector.keys() & right_vector.keys())
    left_norm = math.sqrt(sum(value * value for value in left_vector.values()))
    right_norm = math.sqrt(sum(value * value for value in right_vector.values()))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def weighted_token_overlap(a: Sequence[str], b: Sequence[str],
                           document_frequency: Mapping[str, int], document_count: int) -> float:
    left, right = set(a), set(b)
    if not left or not right:
        return 0.0

    def weight(token: str) -> float:
        return math.log((1.0 + document_count) /
                        (1.0 + document_frequency.get(token, 0))) + 1.0

    union_weight = sum(weight(token) for token in left | right)
    return sum(weight(token) for token in left & right) / union_weight if union_weight else 0.0


def selected_name_ngrams(value: object, max_ngrams: int = 6) -> Set[str]:
    all_ngrams: List[str] = []
    for variant in name_variants(value):
        compact = "".join(variant)
        all_ngrams.extend(compact[i:i + 3] for i in range(max(1, len(compact) - 2)))
    if not all_ngrams:
        return set()
    if max_ngrams <= 0 or len(all_ngrams) <= max_ngrams:
        return set(all_ngrams)
    positions = np.linspace(0, len(all_ngrams) - 1, max_ngrams, dtype=int)
    return {all_ngrams[position] for position in positions}


def _row_value(row: Mapping[str, object] | pd.Series, key: str, default: object = "") -> object:
    if hasattr(row, "_asdict"):
        return row._asdict().get(key, default)
    if isinstance(row, pd.Series):
        if key == "entity_id":
            return row.name
        return row.get(key, default)
    return row.get(key, default)


def build_pair_features(s1_row: Mapping[str, object] | pd.Series,
                        cand_row: Mapping[str, object] | pd.Series,
                        block_index: BlockIndex | None = None,
                        candidate_rank: int | None = None) -> Dict[str, float]:
    profile1 = record_feature_profile(s1_row)
    profile2 = record_feature_profile(cand_row)
    name1_raw, name2_raw = profile1.name_raw, profile2.name_raw
    name1_variants, name2_variants = profile1.name_variants, profile2.name_variants
    name1, name2 = profile1.name_tokens, profile2.name_tokens
    name1_core, name2_core = profile1.name_core, profile2.name_core
    address1_raw, address2_raw = profile1.address_raw, profile2.address_raw
    address1, address2 = profile1.address_tokens, profile2.address_tokens
    country1, country2 = profile1.country, profile2.country
    city1, city2 = profile1.city_tokens, profile2.city_tokens
    state1, state2 = profile1.state_tokens, profile2.state_tokens
    nums1, nums2 = profile1.number_tokens, profile2.number_tokens
    units1, units2 = profile1.unit_numbers, profile2.unit_numbers
    phones1, phones2 = profile1.phone_fragments, profile2.phone_fragments
    postal1, postal2 = profile1.postal_codes, profile2.postal_codes
    layout1, layout2 = profile1.layout_signature, profile2.layout_signature
    house1, house2 = profile1.house_number, profile2.house_number
    landmarks1, landmarks2 = profile1.landmark_tokens, profile2.landmark_tokens

    name_token_j = set_jaccard(name1, name2)
    address_token_j = set_jaccard(address1, address2)
    country_known = bool(country1 and country2)
    same_country = float(country_known and country1 == country2)
    name_char = char_similarity(name1_core, name2_core)
    address_char = char_similarity(address1_raw, address2_raw)
    document_count = block_index.document_count if block_index is not None else 0
    name_df = block_index.name_document_frequency if block_index is not None else {}
    address_df = block_index.address_document_frequency if block_index is not None else {}
    variant_tfidf = max(
        (tfidf_cosine(left, right, name_df, document_count)
         for left in name1_variants for right in name2_variants),
        default=0.0,
    )
    variant_similarity = max(
        (max(set_jaccard(left, right), char_similarity(" ".join(left), " ".join(right)))
         for left in name1_variants for right in name2_variants),
        default=0.0,
    )
    candidate_id = str(_row_value(cand_row, "entity_id"))
    cross_source_features = (
        block_index.target_graph_features.get(candidate_id, {})
        if isinstance(block_index, BlockIndex) else {}
    )
    layout_plausibility: List[float] = []
    postal_plausibility: List[float] = []
    if isinstance(block_index, BlockIndex):
        for country, layout, postal_values in (
            (country1, layout1, postal1), (country2, layout2, postal2),
        ):
            if country:
                layout_plausibility.append(smoothed_profile_frequency(
                    block_index.country_address_layout_counts.get(country, {}), layout
                ))
                postal_counts = block_index.country_postal_format_counts.get(country, {})
                postal_plausibility.extend(
                    smoothed_profile_frequency(postal_counts, len(code)) for code in postal_values
                )

    features = {
        "name_exact": float(bool(name1_raw and name1_raw == name2_raw)),
        "name_core_exact": float(bool(name1_core and name1_core == name2_core)),
        "name_token_jaccard": name_token_j,
        "name_token_containment": containment_similarity(name1, name2),
        "name_char_similarity": name_char,
        "name_jaro_winkler": jaro_winkler_similarity(name1_core, name2_core),
        "name_char_ngram_jaccard": set_jaccard(char_ngrams(name1_core), char_ngrams(name2_core)),
        "name_char_ngram_cosine": char_ngram_cosine(name1_core, name2_core),
        "name_tfidf_cosine": variant_tfidf,
        "name_sorted_token_similarity": char_similarity(" ".join(sorted(name1)), " ".join(sorted(name2))),
        "name_token_order_similarity": token_order_similarity(name1, name2),
        "candidate_retrieval_rank_reciprocal": (
            1.0 / candidate_rank if candidate_rank is not None and candidate_rank > 0 else 0.0
        ),
        "candidate_retrieval_score": (
            9.0 * float(bool(postal1 & postal2))
            + 6.0 * float(bool(house1 and house1 == house2))
            + 3.0 * address_token_j
            + 2.0 * set_jaccard(city1, city2)
            + name_token_j
            - 4.0 * float(country_known and country1 != country2)
        ),
        "contradiction_score": (
            5.0 * float(country_known and country1 != country2)
            + 4.0 * float(bool(postal1 and postal2 and not postal1 & postal2))
            + 4.0 * float(bool(house1 and house2 and house1 != house2))
            + 2.0 * float(bool(state1 and state2 and set(state1) != set(state2)))
        ),
        "name_prefix_similarity": (
            0.0 if not name1_core or not name2_core else
            len(common_prefix(name1_core, name2_core)) / min(len(name1_core), len(name2_core))
        ),
        "name_suffix_similarity": char_similarity(name1_core[::-1], name2_core[::-1]),
        "name_variant_similarity": variant_similarity,
        "name_length_ratio": length_ratio(name1_core, name2_core),
        "common_name_tokens": float(len(set(name1) & set(name2))),
        "name_rare_token_overlap": weighted_token_overlap(name1, name2, name_df, document_count),
        "address_exact": float(bool(address1_raw and address1_raw == address2_raw)),
        "address_token_jaccard": address_token_j,
        "address_token_containment": containment_similarity(address1, address2),
        "address_char_similarity": address_char,
        "address_char_ngram_jaccard": set_jaccard(char_ngrams(address1_raw), char_ngrams(address2_raw)),
        "address_char_ngram_cosine": char_ngram_cosine(address1_raw, address2_raw),
        "address_tfidf_cosine": tfidf_cosine(address1, address2, address_df, document_count),
        "address_length_ratio": length_ratio(address1, address2),
        "common_address_tokens": float(len(set(address1) & set(address2))),
        "address_rare_token_overlap": weighted_token_overlap(address1, address2, address_df, document_count),
        "address_landmark_overlap": set_jaccard(landmarks1, landmarks2),
        "number_token_jaccard": set_jaccard(nums1, nums2),
        "house_number_match": float(bool(house1 and house2 and house1 == house2)),
        "house_number_conflict": float(bool(house1 and house2 and house1 != house2)),
        "unit_number_match": float(bool(units1 and units2 and units1 & units2)),
        "unit_number_conflict": float(bool(units1 and units2 and not units1 & units2)),
        "postal_code_match": float(bool(postal1 & postal2)),
        "postal_code_conflict": float(bool(postal1 and postal2 and not postal1 & postal2)),
        "city_token_jaccard": set_jaccard(city1, city2),
        "city_exact": float(bool(city1 and city2 and set(city1) == set(city2))),
        "state_match": float(bool(state1 and state2 and set(state1) == set(state2))),
        "state_conflict": float(bool(state1 and state2 and set(state1) != set(state2))),
        "tail_component_similarity": set_jaccard(city1[-4:], city2[-4:]),
        "same_country": same_country,
        "country_conflict": float(country_known and country1 != country2),
        "name_address_consistency": name_token_j * address_token_j,
        "city_name_consistency": name_token_j * set_jaccard(city1, city2),
        "name_postal_consistency": name_token_j * float(bool(postal1 & postal2)),
        "phone_fragment_match": float(bool(phones1 & phones2)),
        "cross_source_cluster_size": float(cross_source_features.get("cluster_size", 1.0)),
        "cross_source_support_count": float(cross_source_features.get("support_count", 0.0)),
        "s2_s3_name_similarity": float(cross_source_features.get("name_similarity", 0.0)),
        "s2_s3_address_similarity": float(cross_source_features.get("address_similarity", 0.0)),
        "s2_s3_cluster_agreement": float(cross_source_features.get("cluster_agreement", 0.0)),
        "cross_source_candidate_support_count": 0.0,
        "cross_source_candidate_agreement": 0.0,
        "__candidate_metadata_enriched": 0.0,
        "cross_source_direct_edge_count": float(cross_source_features.get("direct_edge_count", 0.0)),
        "cross_source_max_edge_score": float(cross_source_features.get("max_edge_score", 0.0)),
        "cross_source_min_edge_score": float(cross_source_features.get("min_edge_score", 0.0)),
        "cross_source_avg_edge_score": float(cross_source_features.get("avg_edge_score", 0.0)),
        "candidate_retrieval_channel_count": 0.0,
        "candidate_block_name_phonetic": 0.0,
        "candidate_block_name_tfidf": 0.0,
        "candidate_source_s2": float(candidate_id.startswith("S2-")),
        "candidate_source_s3": float(candidate_id.startswith("S3-")),
        "address_layout_country_plausibility": (
            float(np.mean(layout_plausibility)) if layout_plausibility else 0.0
        ),
        "address_layout_pattern_match": float(bool(
            country_known and country1 == country2 and layout1 == layout2
        )),
        "postal_country_pattern_support": (
            float(np.mean(postal_plausibility)) if postal_plausibility else 0.0
        ),
        "has_name_both": float(bool(name1 and name2)),
        "has_address_both": float(bool(address1 and address2)),
    }
    return features


def common_prefix(a: str, b: str) -> str:
    end = min(len(a), len(b))
    i = 0
    while i < end and a[i] == b[i]:
        i += 1
    return a[:i]


def features_to_vector(feature_dict: Mapping[str, float],
                       feature_names: Sequence[str] = FEATURE_NAMES) -> List[float]:
    return [float(feature_dict.get(name, 0.0)) for name in feature_names]


def _require_candidate_metadata_enriched(feature_rows: Sequence[Mapping[str, float]]) -> None:
    if any(float(row.get("__candidate_metadata_enriched", 1.0)) < 0.5 for row in feature_rows):
        raise ValueError(
            "Candidate pair features must pass through add_candidate_block_memberships before scoring."
        )


def predict_feature_rows(model: Any,
                         feature_rows: Sequence[Mapping[str, float]],
                         feature_names: Sequence[str] = FEATURE_NAMES,
                         batch_size: int = 50000) -> np.ndarray:
    _require_candidate_metadata_enriched(feature_rows)
    probabilities = np.empty(len(feature_rows), dtype=float)
    for start in range(0, len(feature_rows), batch_size):
        end = min(start + batch_size, len(feature_rows))
        matrix = np.asarray(
            [features_to_vector(row, feature_names) for row in feature_rows[start:end]],
            dtype=float,
        )
        probabilities[start:end] = model.predict_proba(matrix)[:, 1]
    return probabilities


def predict_pair_probabilities(model: Any,
                               pair_df: pd.DataFrame,
                               feature_names: Sequence[str] = FEATURE_NAMES,
                               batch_size: int = 50000) -> np.ndarray:
    if pair_df.empty:
        return np.asarray([], dtype=float)
    return predict_feature_rows(
        model, pair_df["features"].tolist(), feature_names, batch_size
    )


def _block_keys_for_values(name: object,
                           address: object,
                           country: object,
                           name_ngram_limit: int = 6) -> Set[Tuple[str, str]]:
    country_key = normalize_country(country)
    name_variants_tokens = name_variants(name)
    addr_tokens = [t for t in address_tokens(address) if len(t) >= 4 and t not in BLOCK_STOPWORDS]
    keys: Set[Tuple[str, str]] = set()

    def add_key(kind: str, value: str) -> None:
        keys.add((kind, f"{country_key}|{value}"))
        # Global copies preserve cross-country recall and provide country-conflict
        # examples when an informative key is shared.
        keys.add((kind, f"*|{value}"))

    for variant in name_variants_tokens:
        name_tokens = [t for t in variant if len(t) >= 3 and t not in BLOCK_STOPWORDS]
        core_name = " ".join(variant)
        if len(core_name) >= 4:
            add_key("name_exact", core_name)
        for token in name_tokens:
            add_key("name_token", token)
        for left, right in zip(name_tokens, name_tokens[1:]):
            add_key("name_pair", f"{left} {right}")
        for token in name_tokens:
            code = soundex_code(token)
            if code:
                add_key("name_phonetic", code)
    for ngram in selected_name_ngrams(name, max_ngrams=name_ngram_limit):
        add_key("name_char_ngram", ngram)
    for token in addr_tokens:
        add_key("address_token", token)
    for code in postal_codes(address):
        add_key("postal", code)
    number = house_number(address)
    if number:
        add_key("house_number", number)
    city = [t for t in address_city_tokens(address, country_key) if len(t) >= 3 and t not in BLOCK_STOPWORDS]
    for token in city:
        add_key("city_token", token)
    return keys


def populate_cross_source_graph(df: pd.DataFrame, index: BlockIndex) -> None:
    """Store direct, verified S2↔S3 edges without taking transitive closure."""
    records: Dict[str, Dict[str, object]] = {}
    name_groups: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(
        lambda: {"S2": [], "S3": []}
    )
    for row in df.itertuples(index=False):
        values = row._asdict()
        entity_id = str(values.get("entity_id", ""))
        source = "S2" if entity_id.startswith("S2-") else "S3" if entity_id.startswith("S3-") else ""
        if not source:
            continue
        records[entity_id] = {
            "entity_id": entity_id,
            "business_name": values.get("business_name", ""),
            "business_address": values.get("business_address", ""),
            "country": values.get("country", ""),
        }
        core_name = " ".join(core_name_tokens(values.get("business_name", "")))
        if core_name:
            name_groups[(normalize_country(values.get("country", "")), core_name)][source].append(entity_id)

    proposed_pairs: Set[Tuple[str, str]] = set()
    signature_pair_limit = 2000
    for (country, _), source_groups in name_groups.items():
        s2_ids, s3_ids = source_groups["S2"], source_groups["S3"]
        if not s2_ids or not s3_ids or len(s2_ids) * len(s3_ids) > 10000:
            continue
        signature_groups: Dict[Tuple[str, str], Dict[str, List[str]]] = defaultdict(
            lambda: {"S2": [], "S3": []}
        )
        for source, entity_ids in (("S2", s2_ids), ("S3", s3_ids)):
            for entity_id in entity_ids:
                row = records[entity_id]
                address = row.get("business_address", "")
                normalized_address = " ".join(address_tokens(address))
                signatures = {("address", normalized_address)} if normalized_address else set()
                signatures.update(("postal", code) for code in postal_codes(address))
                number = house_number(address)
                if number:
                    signatures.add(("house", number))
                for signature in signatures:
                    signature_groups[signature][source].append(entity_id)
        for source_ids in signature_groups.values():
            if (not source_ids["S2"] or not source_ids["S3"] or
                    len(source_ids["S2"]) * len(source_ids["S3"]) > signature_pair_limit):
                continue
            for s2_id in source_ids["S2"]:
                for s3_id in source_ids["S3"]:
                    proposed_pairs.add((s2_id, s3_id))

    edge_by_entity: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for s2_id, s3_id in proposed_pairs:
        features = build_pair_features(records[s2_id], records[s3_id], index)
        strong_address = bool(
            features["address_exact"] or
            (features["postal_code_match"] and features["address_token_jaccard"] >= 0.5) or
            (features["house_number_match"] and features["address_token_jaccard"] >= 0.35)
        )
        if (not strong_address or features["postal_code_conflict"] or
                features["house_number_conflict"] or features["country_conflict"]):
            continue
        name_similarity = float(features["name_variant_similarity"])
        address_similarity = float(features["address_token_jaccard"])
        edge_score = 0.5 * name_similarity + 0.5 * address_similarity
        index.target_graph_neighbors.setdefault(s2_id, set()).add(s3_id)
        index.target_graph_neighbors.setdefault(s3_id, set()).add(s2_id)
        edge_by_entity[s2_id].append((name_similarity, address_similarity, edge_score))
        edge_by_entity[s3_id].append((name_similarity, address_similarity, edge_score))

    for entity_id, edges in edge_by_entity.items():
        scores = [edge[2] for edge in edges]
        opposite_count = len(index.target_graph_neighbors.get(entity_id, set()))
        index.target_graph_features[entity_id] = {
            "cluster_size": float(1 + len(edges)),
            "support_count": float(opposite_count),
            "name_similarity": max(edge[0] for edge in edges),
            "address_similarity": max(edge[1] for edge in edges),
            "cluster_agreement": float(opposite_count > 0),
            "direct_edge_count": float(len(edges)),
            "max_edge_score": max(scores),
            "min_edge_score": min(scores),
            "avg_edge_score": float(np.mean(scores)),
        }


def build_block_index(df: pd.DataFrame,
                      max_block_frequency: int = 10000,
                      name_ngram_limit: int = 6,
                      channel_limit_multiplier: float = 1.0,
                      retrieval_context_mode: str = "once",
                      build_graph: bool = True,
                      semantic_retrieval: bool = False,
                      semantic_max_documents: int = 100000,
                      semantic_top_k: int = 50,
                      semantic_min_similarity: float = 0.12) -> BlockIndex:
    """Build country-aware postings with a block-type-specific frequency ceiling."""
    if name_ngram_limit < 0 or channel_limit_multiplier <= 0:
        raise ValueError("Name n-gram limit cannot be negative and candidate-channel limit must be positive.")
    if retrieval_context_mode not in {"once", "per_channel"}:
        raise ValueError("retrieval_context_mode must be 'once' or 'per_channel'.")
    if semantic_max_documents < 1 or semantic_top_k < 1:
        raise ValueError("Semantic index document and top-K limits must be positive.")
    if not 0.0 <= semantic_min_similarity <= 1.0:
        raise ValueError("Semantic minimum similarity must be between 0 and 1.")
    if semantic_retrieval and len(df) > semantic_max_documents:
        raise ValueError(
            f"Character TF-IDF fallback is capped at {semantic_max_documents:,} target rows; "
            f"received {len(df):,}. Raise --semantic-index-max-targets deliberately or "
            "disable --semantic-retrieval."
        )
    counts: Counter[Tuple[str, str]] = Counter()
    index = BlockIndex()
    index.name_ngram_limit = name_ngram_limit
    index.channel_limits = {
        block_type: max(1, int(math.ceil(limit * channel_limit_multiplier)))
        for block_type, limit in BLOCK_CHANNEL_LIMITS.items()
    }
    index.retrieval_context_mode = retrieval_context_mode
    rows = df.itertuples(index=False)
    for row in rows:
        values = row._asdict()
        index.document_count += 1
        index.name_document_frequency.update({
            token for variant in name_variants(values.get("business_name", "")) for token in variant
        })
        index.address_document_frequency.update(set(address_tokens(values.get("business_address", ""))))
        country_key = normalize_country(values.get("country", ""))
        address = values.get("business_address", "")
        if country_key:
            index.country_address_layout_counts[country_key][
                address_layout_signature(address, country_key)
            ] += 1
            index.country_postal_format_counts[country_key].update(
                len(code) for code in postal_codes(address)
            )
        counts.update(_block_keys_for_values(values.get("business_name", ""),
                                             values.get("business_address", ""),
                                             values.get("country", ""),
                                             name_ngram_limit))

    index.key_document_frequency.update(counts)
    for row in df.itertuples(index=False):
        values = row._asdict()
        entity_id = str(values.get("entity_id", ""))
        keys = _block_keys_for_values(values.get("business_name", ""),
                                      values.get("business_address", ""),
                                      values.get("country", ""),
                                      name_ngram_limit)
        for key in keys:
            kind = key[0]
            adaptive_limit = max_block_frequency * BLOCK_FREQUENCY_MULTIPLIERS.get(kind, 1)
            if max_block_frequency <= 0 or counts[key] <= adaptive_limit:
                index[key].add(entity_id)
    if semantic_retrieval and len(df):
        semantic_documents = [
            " ".join(" ".join(variant) for variant in name_variants(name))
            for name in df["business_name"].astype(str)
        ]
        if any(semantic_documents):
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), min_df=1,
                max_features=100000, sublinear_tf=True, dtype=np.float32,
                norm="l2",
            )
            index.semantic_matrix = vectorizer.fit_transform(semantic_documents)
            index.semantic_vectorizer = vectorizer
            index.semantic_entity_ids = df["entity_id"].astype(str).to_numpy(dtype=object)
            index.semantic_top_k = semantic_top_k
            index.semantic_min_similarity = semantic_min_similarity
    if build_graph:
        populate_cross_source_graph(df, index)
    return index


def candidate_keys_for_record(row: Mapping[str, object] | pd.Series,
                              name_ngram_limit: int = 6) -> Set[Tuple[str, str]]:
    # Word-token retrieval is scored separately with corpus IDF below.
    return {key for key in _block_keys_for_values(
        _row_value(row, "business_name"),
        _row_value(row, "business_address"),
        _row_value(row, "country"),
        name_ngram_limit,
    ) if key[0] not in {"name_token", "address_token", "name_phonetic"}}


def load_source(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")


def parse_truth(path: str) -> Dict[str, Set[str]]:
    truth_df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).fillna("")
    mapping: Dict[str, Set[str]] = {}
    for row in truth_df.itertuples(index=False):
        entity_id = str(getattr(row, "source1_entity_id", ""))
        ids = {value.strip() for value in str(getattr(row, "matched_entity_ids", "")).split(",") if value.strip()}
        mapping[entity_id] = ids
    return mapping


def build_source_lookup(source_df: pd.DataFrame) -> pd.DataFrame:
    """Keep target records in a compact indexed frame instead of millions of Python tuples."""
    return source_df.set_index("entity_id")[["business_name", "business_address", "country"]]


def lookup_record(source_lookup: Mapping[str, object] | pd.DataFrame, entity_id: str) -> object:
    if isinstance(source_lookup, pd.DataFrame):
        return source_lookup.loc[entity_id]
    return source_lookup[entity_id]


def generate_candidate_details(s1_row: Mapping[str, object] | pd.Series,
                               index: Dict[Tuple[str, str], Set[str]],
                               lookup: Mapping[str, object] | pd.DataFrame,
                               collect_blocks: bool = True,
                               ) -> Tuple[List[str], Dict[str, Set[str]]]:
    block_candidates: Dict[str, Set[str]] = defaultdict(set)
    channel_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    doc_count = index.document_count if isinstance(index, BlockIndex) else 0

    def score_postings(kind: str, key: Tuple[str, str], postings: Set[str]) -> None:
        if not postings:
            return
        df = (index.key_document_frequency.get(key, len(postings))
              if isinstance(index, BlockIndex) else len(postings))
        idf = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0
        weight = BLOCK_RETRIEVAL_WEIGHTS.get(kind, 1.0)
        for candidate_id in postings:
            channel_scores[kind][candidate_id] += weight * idf
        if collect_blocks:
            block_candidates[kind].update(postings)

    name_ngram_limit = index.name_ngram_limit if isinstance(index, BlockIndex) else 6
    for key in candidate_keys_for_record(s1_row, name_ngram_limit):
        score_postings(key[0], key, index.get(key, set()))
    if isinstance(index, BlockIndex):
        country = normalize_country(_row_value(s1_row, "country"))
        name_terms = {token for variant in name_variants(_row_value(s1_row, "business_name"))
                      for token in variant if len(token) >= 3 and token not in BLOCK_STOPWORDS}
        address_terms = {token for token in address_tokens(_row_value(s1_row, "business_address"))
                         if len(token) >= 4 and token not in BLOCK_STOPWORDS}
        if collect_blocks:
            block_candidates.setdefault("name_token", set())
            block_candidates.setdefault("address_token", set())

        def retrieve_tfidf(kind: str, terms: Set[str]) -> None:
            for token in terms:
                for scope in (country, "*"):
                    key = (kind, f"{scope}|{token}")
                    score_postings(kind, key, index.get(key, set()))

        retrieve_tfidf("name_token", name_terms)
        retrieve_tfidf("address_token", address_terms)
    symbolic_candidate_ids = set().union(*(set(scores) for scores in channel_scores.values()))
    if len(symbolic_candidate_ids) < PHONETIC_FALLBACK_MIN_SYMBOLIC_CANDIDATES:
        phonetic_keys = {
            key for key in _block_keys_for_values(
                _row_value(s1_row, "business_name"),
                _row_value(s1_row, "business_address"),
                _row_value(s1_row, "country"),
                index.name_ngram_limit if isinstance(index, BlockIndex) else 6,
            ) if key[0] == "name_phonetic"
        }
        for key in phonetic_keys:
            score_postings("name_phonetic", key, index.get(key, set()))
    valid_ids = lookup.index if isinstance(lookup, pd.DataFrame) else lookup
    fallback_candidate_ids = set().union(*(set(scores) for scores in channel_scores.values()))
    if (isinstance(index, BlockIndex) and index.semantic_vectorizer is not None
            and index.semantic_matrix is not None
            and len(fallback_candidate_ids) < PHONETIC_FALLBACK_MIN_SYMBOLIC_CANDIDATES):
        query_document = " ".join(
            " ".join(variant)
            for variant in name_variants(_row_value(s1_row, "business_name"))
        )
        query_vector = index.semantic_vectorizer.transform([query_document])
        if query_vector.nnz:
            similarities = (index.semantic_matrix @ query_vector.T).tocoo()
            if similarities.nnz > index.semantic_top_k:
                top_positions = np.argpartition(
                    similarities.data, -index.semantic_top_k
                )[-index.semantic_top_k:]
            else:
                top_positions = np.arange(similarities.nnz)
            ranked_semantic = sorted(
                ((int(similarities.row[position]), float(similarities.data[position]))
                 for position in top_positions),
                key=lambda item: (-item[1], str(index.semantic_entity_ids[item[0]])),
            )
            semantic_candidates: Set[str] = set()
            for document_position, similarity in ranked_semantic:
                candidate_id = str(index.semantic_entity_ids[document_position])
                if similarity < index.semantic_min_similarity:
                    continue
                if candidate_id not in valid_ids or not candidate_id.startswith(("S2-", "S3-")):
                    continue
                semantic_candidates.add(candidate_id)
                channel_scores["name_tfidf"][candidate_id] = (
                    BLOCK_RETRIEVAL_WEIGHTS["name_tfidf"] * float(similarity)
                )
                if collect_blocks:
                    block_candidates["name_tfidf"].add(candidate_id)
                if len(semantic_candidates) >= index.semantic_top_k:
                    break
            if collect_blocks:
                block_candidates["name_tfidf"].update(semantic_candidates)
    query_name = set(core_name_tokens(_row_value(s1_row, "business_name")))
    query_address = set(address_tokens(_row_value(s1_row, "business_address")))
    query_city = set(address_city_tokens(
        _row_value(s1_row, "business_address"), _row_value(s1_row, "country")
    ))
    query_postal = postal_codes(_row_value(s1_row, "business_address"))
    query_house = house_number(_row_value(s1_row, "business_address"))
    query_country = normalize_country(_row_value(s1_row, "country"))
    context_scores: Dict[str, float] = {}
    all_channel_candidate_ids = set().union(*(set(scores) for scores in channel_scores.values()))
    for candidate_id in all_channel_candidate_ids:
        if candidate_id not in valid_ids:
            continue
        candidate_row = lookup_record(lookup, candidate_id)
        candidate_address = candidate_row.get("business_address", "")
        candidate_country = normalize_country(candidate_row.get("country", ""))
        candidate_name = set(core_name_tokens(candidate_row.get("business_name", "")))
        candidate_address_tokens = set(address_tokens(candidate_address))
        candidate_city = set(address_city_tokens(candidate_address, candidate_country))
        candidate_postal = postal_codes(candidate_address)
        candidate_house = house_number(candidate_address)
        retrieval_score = (
            9.0 * float(bool(query_postal & candidate_postal))
            + 6.0 * float(bool(query_house and query_house == candidate_house))
            + 3.0 * set_jaccard(query_address, candidate_address_tokens)
            + 2.0 * set_jaccard(query_city, candidate_city)
            + 1.0 * set_jaccard(query_name, candidate_name)
        )
        if query_country and candidate_country and query_country != candidate_country:
            retrieval_score -= 4.0
        context_scores[candidate_id] = retrieval_score

    candidate_scores: Dict[str, float] = defaultdict(float)
    candidate_channel_counts: Counter[str] = Counter()
    selected_by_channel: Dict[str, Set[str]] = {}
    for block_type, scores in channel_scores.items():
        valid_scores = {eid: score for eid, score in scores.items()
                        if eid.startswith(("S2-", "S3-")) and eid in valid_ids}
        ranked = sorted(valid_scores, key=lambda eid: (
            -(valid_scores[eid] + context_scores.get(eid, 0.0)), eid
        ))
        channel_limits = (index.channel_limits if isinstance(index, BlockIndex) else BLOCK_CHANNEL_LIMITS)
        selected = set(ranked[:channel_limits.get(block_type, len(ranked))])
        selected_by_channel[block_type] = selected
        for candidate_id in selected:
            candidate_channel_counts[candidate_id] += 1
            candidate_scores[candidate_id] += valid_scores[candidate_id]
            if not isinstance(index, BlockIndex) or index.retrieval_context_mode == "per_channel":
                candidate_scores[candidate_id] += context_scores.get(candidate_id, 0.0)
    if isinstance(index, BlockIndex) and index.retrieval_context_mode == "once":
        for candidate_id in candidate_scores:
            candidate_scores[candidate_id] += context_scores.get(candidate_id, 0.0)
            candidate_scores[candidate_id] += 0.5 * math.log1p(
                max(0, candidate_channel_counts[candidate_id] - 1)
            )
    candidate_ids = set(candidate_scores)
    if collect_blocks:
        for block_type in BLOCK_TYPE_NAMES:
            block_candidates[block_type] = selected_by_channel.get(block_type, set())
    ordered = sorted(candidate_ids, key=lambda eid: (-candidate_scores[eid], eid))
    return ordered, dict(block_candidates) if collect_blocks else {}


def generate_candidates(s1_row: Mapping[str, object] | pd.Series,
                        index: Dict[Tuple[str, str], Set[str]],
                        lookup: Mapping[str, object] | pd.DataFrame) -> List[str]:
    return generate_candidate_details(s1_row, index, lookup, collect_blocks=False)[0]


def add_candidate_block_memberships(feature_rows: List[Dict[str, float]],
                                    candidate_ids: Sequence[str],
                                    block_candidates: Mapping[str, Set[str]],
                                    index: BlockIndex | None = None) -> None:
    positions_by_candidate = {candidate_id: position
                              for position, candidate_id in enumerate(candidate_ids)}
    for block_type, block_ids in block_candidates.items():
        for candidate_id in block_ids:
            position = positions_by_candidate.get(candidate_id)
            if position is not None:
                feature_rows[position][f"candidate_block_{block_type}"] = 1.0
    for candidate_id, position in positions_by_candidate.items():
        feature_rows[position]["__candidate_metadata_enriched"] = 1.0
        channel_count = sum(candidate_id in block_ids for block_ids in block_candidates.values())
        feature_rows[position]["candidate_retrieval_channel_count"] = float(channel_count)
        if not isinstance(index, BlockIndex):
            continue
        direct_peers = index.target_graph_neighbors.get(candidate_id, set()) & positions_by_candidate.keys()
        source_prefix = "S3-" if candidate_id.startswith("S2-") else "S2-"
        opposite_source_peers = sum(peer.startswith(source_prefix) for peer in direct_peers)
        feature_rows[position]["cross_source_candidate_support_count"] = float(len(direct_peers))
        feature_rows[position]["cross_source_candidate_agreement"] = float(opposite_source_peers > 0)


def candidate_recall(candidate_map: Mapping[str, Set[str]], truth: Mapping[str, Set[str]]) -> float:
    total_truth = sum(len(truth.get(entity_id, set())) for entity_id in candidate_map)
    if total_truth == 0:
        return 1.0
    found = sum(len(candidates & truth.get(entity_id, set()))
                for entity_id, candidates in candidate_map.items())
    return found / total_truth


def histogram_percentile(histogram: Mapping[int, int], percentile: float) -> float:
    count = sum(histogram.values())
    if not count:
        return 0.0
    target = max(0, int(math.ceil(percentile * count)) - 1)
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative > target:
            return float(value)
    return float(max(histogram))


def print_candidate_diagnostics(label: str,
                                diagnostics: CandidateDiagnostics,
                                pair_recall: float) -> None:
    histogram = diagnostics.candidate_size_histogram
    count = diagnostics.entity_count
    average = (sum(size * frequency for size, frequency in histogram.items()) / count
               if count else 0.0)
    maximum = max(histogram, default=0)
    p50 = histogram_percentile(histogram, 0.50)
    p95 = histogram_percentile(histogram, 0.95)
    p99 = histogram_percentile(histogram, 0.99)
    print(f"{label} candidates/entity: mean={average:.1f}, p50={p50:.0f}, p95={p95:.0f}, "
          f"p99={p99:.0f}, max={maximum}; pair recall={pair_recall:.4f}; "
          f"example-build={diagnostics.example_build_seconds:.1f}s")

    if diagnostics.total_truth_pairs:
        for limit in CANDIDATE_EFFICIENCY_K:
            recall_at_limit = diagnostics.recall_at_k_hits[limit] / diagnostics.total_truth_pairs
            average_at_limit = diagnostics.candidates_at_k[limit] / max(count, 1)
            print(f"  diagnostic retrieval@{limit}: recall={recall_at_limit:.4f}, "
                  f"mean candidates kept={average_at_limit:.1f}")

    for block_type in BLOCK_TYPE_NAMES:
        block_histogram = diagnostics.block_size_histograms[block_type]
        block_count = sum(block_histogram.values())
        block_mean = (sum(size * frequency for size, frequency in block_histogram.items()) / block_count
                      if block_count else 0.0)
        block_recall = (diagnostics.block_truth_hits[block_type] / diagnostics.total_truth_pairs
                        if diagnostics.total_truth_pairs else 1.0)
        block_maximum = max(block_histogram, default=0)
        block_p50 = histogram_percentile(block_histogram, 0.50)
        block_p95 = histogram_percentile(block_histogram, 0.95)
        block_p99 = histogram_percentile(block_histogram, 0.99)
        print(f"  block {block_type}: mean={block_mean:.1f}, p50={block_p50:.0f}, "
              f"p95={block_p95:.0f}, p99={block_p99:.0f}, max={block_maximum}, "
          f"truth recall={block_recall:.4f}")


def run_retrieval_ablation(source1: pd.DataFrame,
                           truth: Mapping[str, Set[str]],
                           target_sources: pd.DataFrame,
                           source_lookup: pd.DataFrame,
                           max_block_frequency: int,
                           current_ngram_limit: int,
                           current_channel_multiplier: float,
                           current_context_mode: str,
                           output_path: Path) -> pd.DataFrame:
    """Measure recall, list size, and runtime over retrieval profile combinations."""
    profiles = {
        (ngram_limit, multiplier, context_mode)
        for ngram_limit in (6, 10, 16, 24, 32, 0)
        for multiplier in (0.75, 1.0, 1.5, 2.0)
        for context_mode in ("once", "per_channel")
    }
    profiles.add((current_ngram_limit, current_channel_multiplier, current_context_mode))
    lookup = source_lookup
    rows: List[Dict[str, float]] = []
    for ngram_limit, channel_multiplier, context_mode in sorted(profiles):
        started = time.perf_counter()
        index = build_block_index(
            target_sources,
            max_block_frequency=max_block_frequency,
            name_ngram_limit=ngram_limit,
            channel_limit_multiplier=channel_multiplier,
            retrieval_context_mode=context_mode,
            build_graph=False,
        )
        index_build_seconds = time.perf_counter() - started
        query_started = time.perf_counter()
        size_histogram: Counter[int] = Counter()
        recall_hits: Counter[int] = Counter()
        total_truth_pairs = 0
        found_truth_pairs = 0
        query_count = 0
        for row in source1.itertuples(index=False):
            values = row._asdict()
            entity_id = str(values.get("entity_id", ""))
            candidates = generate_candidate_details(
                values, index, lookup, collect_blocks=False
            )[0]
            true_ids = truth.get(entity_id, set())
            total_truth_pairs += len(true_ids)
            found_truth_pairs += len(set(candidates) & true_ids)
            size_histogram[len(candidates)] += 1
            for limit in CANDIDATE_EFFICIENCY_K:
                recall_hits[limit] += len(set(candidates[:limit]) & true_ids)
            query_count += 1
        query_seconds = time.perf_counter() - query_started
        row: Dict[str, Any] = {
            "name_ngram_limit": "all" if ngram_limit == 0 else ngram_limit,
            "channel_limit_multiplier": float(channel_multiplier),
            "retrieval_context_mode": context_mode,
            "entity_count": float(query_count),
            "candidate_mean": (
                sum(size * count for size, count in size_histogram.items()) / query_count
                if query_count else 0.0
            ),
            "candidate_p50": histogram_percentile(size_histogram, 0.50),
            "candidate_p95": histogram_percentile(size_histogram, 0.95),
            "candidate_p99": histogram_percentile(size_histogram, 0.99),
            "candidate_max": float(max(size_histogram, default=0)),
            "candidate_recall": found_truth_pairs / total_truth_pairs if total_truth_pairs else 1.0,
            "index_build_seconds": index_build_seconds,
            "query_seconds": query_seconds,
            "total_seconds": index_build_seconds + query_seconds,
        }
        for limit in CANDIDATE_EFFICIENCY_K:
            row[f"recall_at_{limit}"] = (
                recall_hits[limit] / total_truth_pairs if total_truth_pairs else 1.0
            )
        rows.append(row)
        ngram_label = "all" if ngram_limit == 0 else str(ngram_limit)
        print(f"Retrieval ngrams={ngram_label} channel-x{channel_multiplier:g} "
              f"context={context_mode}: "
              f"recall={row['candidate_recall']:.4f}, mean={row['candidate_mean']:.1f}, "
              f"p95={row['candidate_p95']:.0f}, p99={row['candidate_p99']:.0f}, "
              f"runtime={row['total_seconds']:.1f}s")
        del index
        gc.collect()
    result = pd.DataFrame(rows).sort_values(
        ["candidate_recall", "candidate_mean"], ascending=[False, True]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def negative_hardness(features: Mapping[str, float]) -> float:
    return (
        1.5 * features["name_char_similarity"]
        + 1.1 * features["name_token_jaccard"]
        + 0.8 * features["name_tfidf_cosine"]
        + 0.8 * features["address_token_jaccard"]
        + 0.5 * features["address_tfidf_cosine"]
        + 0.5 * features["city_token_jaccard"]
        + 0.4 * features["postal_code_match"]
        + 0.3 * features["same_country"]
    )


def negative_category(features: Mapping[str, float]) -> str:
    if features["name_core_exact"] and features["address_token_jaccard"] < 0.5:
        return "same_name_wrong_address"
    if (features["address_exact"] or features["address_token_jaccard"] >= 0.8) and features["name_token_jaccard"] < 0.5:
        return "same_address_wrong_name"
    if features["city_token_jaccard"] > 0:
        return "same_city"
    if features["same_country"]:
        return "same_country"
    return "other"


def mixed_negative_pool_counts(total: int) -> Tuple[int, int, int, int]:
    """Return hard/medium/structural/random counts with deterministic rounding."""
    if total <= 0:
        return 0, 0, 0, 0
    ratios = (0.40, 0.30, 0.20, 0.10)
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    tie_priority = (3, 1, 2, 0)  # keep a random tail when integer rounding permits
    for position in sorted(range(4), key=lambda item: (-(raw[item] - counts[item]),
                                                        tie_priority.index(item)))[
            :total - sum(counts)]:
        counts[position] += 1
    if total >= 4:
        for position in (1, 2, 3):
            if counts[position] == 0:
                donor = next((candidate for candidate in (0, 1, 2)
                              if counts[candidate] > 1), None)
                if donor is not None:
                    counts[donor] -= 1
                    counts[position] += 1
    return tuple(counts)  # type: ignore[return-value]


def select_mixed_hard_negatives(ranked: Sequence[Tuple[float, str, str]],
                                limit: int,
                                rng: np.random.Generator) -> List[str]:
    """Sample 40/30/20/10 hardest, medium, structural, and random negatives."""
    ordered = sorted(ranked, key=lambda item: (-float(item[0]), str(item[1])))
    target = min(max(0, int(limit)), len(ordered))
    if not target:
        return []
    hard_count, medium_count, structural_count, random_count = mixed_negative_pool_counts(target)
    selected: List[str] = []
    selected_ids: Set[str] = set()

    def add(items: Sequence[Tuple[float, str, str]], count: int,
            randomize: bool = False) -> None:
        available = [item for item in items if item[1] not in selected_ids]
        if randomize and len(available) > count:
            chosen_positions = sorted(rng.choice(len(available), size=count, replace=False).tolist())
            available = [available[position] for position in chosen_positions]
        for _, candidate_id, _ in available[:count]:
            if candidate_id not in selected_ids:
                selected.append(candidate_id)
                selected_ids.add(candidate_id)

    add(ordered, hard_count)
    low = int(math.floor(len(ordered) * 0.30))
    high = max(low + 1, int(math.ceil(len(ordered) * 0.80)))
    add(ordered[low:high], medium_count, randomize=True)
    structural = [item for item in ordered
                  if item[2] in {"same_name_wrong_address", "same_address_wrong_name", "same_city"}]
    add(structural, structural_count)
    add(ordered, random_count, randomize=True)
    if len(selected) < target:
        add(ordered, target - len(selected))
    return selected[:target]


def build_training_examples(train_source1: pd.DataFrame,
                            source2_source3_df: pd.DataFrame | None,
                            truth: Mapping[str, Set[str]],
                            max_s1: int | None = None,
                            negatives_per_positive: int = 20,
                            random_negatives: int = 5,
                            training: bool = True,
                            mine_hard_negatives: bool = True,
                            seed: int = 42,
                            source_lookup: Mapping[str, object] | pd.DataFrame | None = None,
                            index: Dict[Tuple[str, str], Set[str]] | None = None
                            ) -> Tuple[pd.DataFrame, float, CandidateDiagnostics]:
    if source_lookup is None and source2_source3_df is not None:
        source_lookup = build_source_lookup(source2_source3_df)
    if index is None and source2_source3_df is not None:
        index = build_block_index(source2_source3_df)
    if source_lookup is None or index is None:
        raise ValueError("Provide the Source-2/3 lookup and block index, or the source dataframe.")

    examples: List[Dict[str, object]] = []
    total_truth_pairs = 0
    found_truth_pairs = 0
    diagnostics = CandidateDiagnostics()
    build_started = time.perf_counter()
    s1_rows = train_source1
    if max_s1 is not None and max_s1 >= 0 and len(s1_rows) > max_s1:
        s1_rows = s1_rows.sample(n=max_s1, random_state=seed)

    for row in s1_rows.itertuples(index=False):
        s1_values = row._asdict()
        entity_id = str(s1_values.get("entity_id", ""))
        candidates, block_candidates = generate_candidate_details(s1_values, index, source_lookup)
        candidate_set = set(candidates)
        candidate_ranks = {candidate_id: position + 1
                           for position, candidate_id in enumerate(candidates)}
        candidate_metadata: List[Dict[str, float]] = [dict() for _ in candidates]
        add_candidate_block_memberships(
            candidate_metadata, candidates, block_candidates,
            index if isinstance(index, BlockIndex) else None,
        )
        def candidate_features(candidate_id: str) -> Dict[str, float]:
            features = build_pair_features(
                s1_values, lookup_record(source_lookup, candidate_id), index,
                candidate_rank=candidate_ranks[candidate_id],
            )
            features.update(candidate_metadata[candidate_ranks[candidate_id] - 1])
            return features

        true_ids = truth.get(entity_id, set())
        diagnostics.entity_count += 1
        diagnostics.total_truth_pairs += len(true_ids)
        diagnostics.found_truth_pairs += len(candidate_set & true_ids)
        diagnostics.candidate_size_histogram[len(candidates)] += 1
        for block_type in BLOCK_TYPE_NAMES:
            block_ids = block_candidates.get(block_type, set())
            diagnostics.block_size_histograms[block_type][len(block_ids)] += 1
            diagnostics.block_truth_hits[block_type] += len(block_ids & true_ids)
        for limit in CANDIDATE_EFFICIENCY_K:
            diagnostics.recall_at_k_hits[limit] += len(set(candidates[:limit]) & true_ids)
            diagnostics.candidates_at_k[limit] += min(limit, len(candidates))
        total_truth_pairs += len(true_ids)
        positive_ids = candidate_set & true_ids
        found_truth_pairs += len(positive_ids)
        negative_ids = sorted(candidate_set - true_ids)

        if training and len(negative_ids) > 0:
            target_negatives = min(
                len(negative_ids),
                max(negatives_per_positive, negatives_per_positive * max(1, len(positive_ids))),
            )
            stable_seed = int(hashlib.sha1(f"{seed}:{entity_id}".encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(stable_seed)
            if mine_hard_negatives:
                ranked: List[Tuple[float, str, str]] = []
                for candidate_id in negative_ids:
                    features = candidate_features(candidate_id)
                    ranked.append((negative_hardness(features), candidate_id, negative_category(features)))
                selected_negative_ids = select_mixed_hard_negatives(
                    ranked, target_negatives, rng
                )
            else:
                selected_positions = rng.choice(len(negative_ids), size=target_negatives, replace=False)
                selected_negative_ids = [negative_ids[position] for position in sorted(selected_positions)]
            negative_features = [
                (candidate_id, candidate_features(candidate_id))
                for candidate_id in selected_negative_ids
            ]
        else:
            negative_features = [
                (candidate_id, candidate_features(candidate_id))
                for candidate_id in negative_ids
            ]

        for candidate_id in sorted(positive_ids):
            examples.append({
                "entity_id": entity_id,
                "candidate_id": candidate_id,
                "label": 1,
                "features": candidate_features(candidate_id),
            })
        for candidate_id, features in negative_features:
            examples.append({
                "entity_id": entity_id,
                "candidate_id": candidate_id,
                "label": 0,
                "features": features,
            })

    frame = pd.DataFrame(examples, columns=["entity_id", "candidate_id", "label", "features"])
    diagnostics.example_build_seconds = time.perf_counter() - build_started
    recall = found_truth_pairs / total_truth_pairs if total_truth_pairs else 1.0
    return frame, recall, diagnostics


def split_training_data(train_source1: pd.DataFrame,
                        validation_fraction: float = 0.2,
                        random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = train_source1.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split = int(len(shuffled) * (1.0 - validation_fraction))
    return shuffled.iloc[:split].copy(), shuffled.iloc[split:].copy()


def stratified_sample_source1(source1: pd.DataFrame,
                              truth: Mapping[str, Set[str]],
                              sample_size: int,
                              name_frequencies: Mapping[str, int],
                              seed: int) -> pd.DataFrame:
    """Sample linked/singleton and common/rare-name entities proportionally."""
    if sample_size <= 0 or len(source1) <= sample_size:
        return source1.copy()
    strata: Dict[Tuple[bool, bool], List[int]] = defaultdict(list)
    for position, row in enumerate(source1.itertuples(index=False)):
        values = row._asdict()
        entity_id = str(values.get("entity_id", ""))
        name = " ".join(core_name_tokens(values.get("business_name", "")))
        repeated_name = bool(name and name_frequencies.get(name, 0) > 1)
        strata[(bool(truth.get(entity_id, set())), repeated_name)].append(position)

    allocations = {key: 0 for key in strata}
    if sample_size >= len(strata):
        for key in strata:
            allocations[key] = 1
        budget = sample_size - len(strata)
        capacity_total = sum(len(positions) - 1 for positions in strata.values())
        if capacity_total and budget:
            raw_additions = {key: budget * (len(positions) - 1) / capacity_total
                             for key, positions in strata.items()}
            additions = {key: int(amount) for key, amount in raw_additions.items()}
            for key, amount in additions.items():
                allocations[key] += amount
            remaining = budget - sum(additions.values())
            largest_remainders = sorted(
                strata,
                key=lambda key: (-(raw_additions[key] - additions[key]), key),
            )
            for key in largest_remainders:
                if remaining <= 0:
                    break
                if allocations[key] < len(strata[key]):
                    allocations[key] += 1
                    remaining -= 1
    else:
        for key in sorted(strata, key=lambda value: (-len(strata[value]), value))[:sample_size]:
            allocations[key] = 1

    selected: List[int] = []
    for stratum_number, key in enumerate(sorted(strata)):
        quota = min(allocations[key], len(strata[key]))
        if quota:
            rng = np.random.default_rng(seed + stratum_number)
            selected.extend(rng.choice(strata[key], size=quota, replace=False).tolist())
    sampled = source1.iloc[selected]
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def score_entity_predictions(predicted_matches: Mapping[str, Set[str]],
                             truth: Mapping[str, Set[str]],
                             entity_ids: Iterable[str] | None = None) -> float:
    ids = (list(dict.fromkeys(str(value) for value in entity_ids))
           if entity_ids is not None else sorted(set(predicted_matches) | set(truth)))
    if not ids:
        return 0.0
    scores: List[float] = []
    for entity_id in ids:
        pred = predicted_matches.get(entity_id, set())
        expected = truth.get(entity_id, set())
        tp = len(pred & expected)
        fp = len(pred - expected)
        fn = len(expected - pred)
        denominator = 1.25 * tp + fp + 0.25 * fn
        scores.append(1.0 if denominator == 0 else (1.25 * tp) / denominator)
    return float(np.mean(scores))


def threshold_search(model: Any,
                     val_df: pd.DataFrame,
                     truth: Mapping[str, Set[str]],
                     validation_entity_ids: Sequence[str],
                     feature_names: Sequence[str] = FEATURE_NAMES) -> Tuple[float, float | None, float]:
    probabilities = predict_pair_probabilities(model, val_df, feature_names)
    return threshold_search_from_scores(val_df, probabilities, truth, validation_entity_ids)


def threshold_search_from_scores(pair_df: pd.DataFrame,
                                 probabilities: np.ndarray,
                                 truth: Mapping[str, Set[str]],
                                 validation_entity_ids: Sequence[str]) -> Tuple[float, float | None, float]:
    """Find exact score decision boundaries with a vectorized event sweep."""
    entity_ids = [str(value) for value in validation_entity_ids]
    entity_to_pos = {entity_id: i for i, entity_id in enumerate(entity_ids)}
    true_counts = np.asarray([len(truth.get(entity_id, set())) for entity_id in entity_ids], dtype=np.int64)

    if pair_df.empty or not len(probabilities):
        return 1.000001, None, score_entity_predictions({}, truth, entity_ids)

    labels = pair_df["label"].astype(int).to_numpy()
    pair_entity_positions = pair_df["entity_id"].astype(str).map(entity_to_pos).to_numpy(dtype=np.int64)
    entity_max = np.full(len(entity_ids), -np.inf, dtype=float)
    np.maximum.at(entity_max, pair_entity_positions, probabilities)

    best_threshold = 0.0
    best_margin: float | None = None
    best_score = -1.0
    margin_options: Sequence[float | None] = (0.05, 0.1, 0.2, 0.35, 0.5, None)
    entity_sort = np.lexsort((probabilities, pair_entity_positions))
    sorted_entity_positions = pair_entity_positions[entity_sort]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_entity_positions)) + 1]
    group_ends = np.r_[group_starts[1:], len(entity_sort)]
    for margin in margin_options:
        floor = (np.zeros(len(entity_ids), dtype=float) if margin is None else
                 np.maximum(0.0, entity_max - margin))
        initial_total = 0.0
        event_thresholds: List[float] = []
        event_deltas: List[float] = []
        seen_entities = np.zeros(len(entity_ids), dtype=bool)
        for entity_position, start, end in zip(
            sorted_entity_positions[group_starts], group_starts, group_ends
        ):
            seen_entities[entity_position] = True
            positions = entity_sort[start:end]
            entity_probabilities = probabilities[positions]
            entity_labels = labels[positions]
            selected = entity_probabilities >= floor[entity_position]
            eligible_scores = entity_probabilities[selected]
            eligible_labels = entity_labels[selected]
            predicted_count = len(eligible_scores)
            true_positive_count = int(eligible_labels.sum())
            truth_count = int(true_counts[entity_position])

            def entity_score(tp: int, predicted: int) -> float:
                denominator = 1.25 * tp + (predicted - tp) + 0.25 * (truth_count - tp)
                return 1.0 if denominator == 0 else (1.25 * tp) / denominator

            current_score = entity_score(true_positive_count, predicted_count)
            initial_total += current_score
            if not predicted_count:
                continue
            score_order = np.argsort(eligible_scores, kind="stable")
            sorted_scores = eligible_scores[score_order]
            sorted_labels = eligible_labels[score_order]
            unique_scores, starts, counts = np.unique(
                sorted_scores, return_index=True, return_counts=True
            )
            true_counts_by_score = np.add.reduceat(sorted_labels.astype(np.int64), starts)
            for candidate_score, removed_count, removed_true_count in zip(
                unique_scores, counts, true_counts_by_score
            ):
                next_score = entity_score(
                    true_positive_count - int(removed_true_count),
                    predicted_count - int(removed_count),
                )
                delta = next_score - current_score
                event_thresholds.append(float(np.nextafter(candidate_score, np.inf)))
                event_deltas.append(delta)
                true_positive_count -= int(removed_true_count)
                predicted_count -= int(removed_count)
                current_score = next_score

        # Keep zero-delta boundaries too, so conservative tie-breaking can
        # choose any exact score boundary even when a margin already excludes it.
        event_thresholds.extend(
            float(np.nextafter(candidate_score, np.inf))
            for candidate_score in np.unique(probabilities)
        )
        event_deltas.extend(0.0 for _ in np.unique(probabilities))

        initial_total += float(np.count_nonzero(~seen_entities & (true_counts == 0)))

        base_score = initial_total / max(len(entity_ids), 1)
        candidate_thresholds = [0.0]
        candidate_scores = [base_score]
        if event_thresholds:
            unique_thresholds, inverse = np.unique(event_thresholds, return_inverse=True)
            deltas = np.bincount(inverse, weights=np.asarray(event_deltas, dtype=float))
            candidate_thresholds.extend(unique_thresholds.tolist())
            candidate_scores.extend((base_score + np.cumsum(deltas) / max(len(entity_ids), 1)).tolist())

        for threshold, score in zip(candidate_thresholds, candidate_scores):
            more_conservative_tie = (
                np.isclose(score, best_score)
                and (threshold > best_threshold or (
                    threshold == best_threshold
                    and margin is not None
                    and (best_margin is None or margin < best_margin)
                ))
            )
            score_is_better = score > best_score and not np.isclose(score, best_score)
            if score_is_better or more_conservative_tie:
                best_score = float(score)
                best_threshold = float(threshold)
                best_margin = margin
    return best_threshold, best_margin, best_score


def entity_decision_feature_vector(feature_rows: Sequence[Mapping[str, float]],
                                   probabilities: np.ndarray,
                                   pair_threshold: float) -> List[float]:
    if not len(probabilities):
        return [0.0] * len(ENTITY_FEATURE_NAMES)
    order = np.argsort(-probabilities, kind="stable")
    top = int(order[0])
    second_probability = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    ranked_position = np.empty(len(order), dtype=np.int64)
    ranked_position[order] = np.arange(1, len(order) + 1)
    name_scores = np.asarray([float(row.get("name_variant_similarity", 0.0)) for row in feature_rows])
    address_scores = np.asarray([float(row.get("address_token_jaccard", 0.0)) for row in feature_rows])
    postal_scores = np.asarray([float(row.get("postal_code_match", 0.0)) for row in feature_rows])
    house_scores = np.asarray([float(row.get("house_number_match", 0.0)) for row in feature_rows])
    best_name_position = int(np.argmax(name_scores))
    best_address_position = int(np.argmax(address_scores))
    best_probability = float(probabilities[order[0]])
    top2_probability = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    top3_probability = float(probabilities[order[2]]) if len(order) > 2 else 0.0
    s2_scores = [float(probabilities[position]) for position, row in enumerate(feature_rows)
                 if float(row.get("candidate_source_s2", 0.0)) > 0]
    s3_scores = [float(probabilities[position]) for position, row in enumerate(feature_rows)
                 if float(row.get("candidate_source_s3", 0.0)) > 0]
    block_fields = (
        ("name_exact", "name_exact"), ("name_pair", "name_pair"),
        ("name_char_ngram", "name_char_ngram"), ("name_phonetic", "name_phonetic"),
        ("name_tfidf", "name_tfidf"),
        ("name_token", "name_token"),
        ("address_token", "address_token"), ("postal", "postal"),
        ("house_number", "house_number"), ("city_token", "city_token"),
    )
    block_counts = [
        float(sum(float(row.get(f"candidate_block_{key}", 0.0)) for row in feature_rows))
        for key, _ in block_fields
    ]
    clipped_top = float(np.clip(best_probability, 1e-6, 1.0 - 1e-6))
    clipped_second = float(np.clip(top2_probability, 1e-6, 1.0 - 1e-6))
    best_logit = math.log(clipped_top / (1.0 - clipped_top))
    second_logit = math.log(clipped_second / (1.0 - clipped_second))
    return [
        best_probability,
        second_probability,
        best_probability - second_probability,
        float(np.mean(probabilities[order[:min(3, len(order))]])),
        float(np.std(probabilities)),
        float(math.log1p(len(probabilities))),
        float(np.count_nonzero(probabilities >= pair_threshold)),
        float(name_scores[best_name_position]),
        float(address_scores[best_address_position]),
        float(feature_rows[top].get("same_country", 0.0)),
        float(1.0 / ranked_position[best_name_position]),
        float(1.0 / ranked_position[best_address_position]),
        float(feature_rows[top].get("candidate_retrieval_rank_reciprocal", 0.0)),
        best_logit - second_logit,
        best_probability / max(top2_probability, 1e-6),
        best_probability - top3_probability,
        float(name_scores[order[0]]),
        float(name_scores[order[1]]) if len(order) > 1 else 0.0,
        float(address_scores[order[0]]),
        float(address_scores[order[1]]) if len(order) > 1 else 0.0,
        float(postal_scores[order[0]]),
        float(postal_scores[order[1]]) if len(order) > 1 else 0.0,
        float(house_scores[order[0]]),
        float(house_scores[order[1]]) if len(order) > 1 else 0.0,
        float(sum(float(row.get("name_core_exact", 0.0)) for row in feature_rows)),
        float(sum(float(row.get("address_exact", 0.0)) for row in feature_rows)),
        float(sum(float(row.get("postal_code_match", 0.0)) for row in feature_rows)),
        *block_counts,
        float(sum(float(row.get("candidate_source_s2", 0.0)) for row in feature_rows)),
        float(sum(float(row.get("candidate_source_s3", 0.0)) for row in feature_rows)),
        max(s2_scores, default=0.0),
        max(s3_scores, default=0.0),
        max(s2_scores, default=0.0) - max(s3_scores, default=0.0),
        float(feature_rows[order[0]].get("country_conflict", 0.0)),
        float(feature_rows[order[1]].get("country_conflict", 0.0)) if len(order) > 1 else 0.0,
        float(feature_rows[order[0]].get("postal_code_conflict", 0.0)),
        float(feature_rows[order[1]].get("postal_code_conflict", 0.0)) if len(order) > 1 else 0.0,
        float(feature_rows[order[0]].get("house_number_conflict", 0.0)),
        float(feature_rows[order[1]].get("house_number_conflict", 0.0)) if len(order) > 1 else 0.0,
        float(feature_rows[order[0]].get("name_core_exact", 0.0)),
        float(feature_rows[order[1]].get("name_core_exact", 0.0)) if len(order) > 1 else 0.0,
        float(feature_rows[order[0]].get("cross_source_candidate_support_count", 0.0)),
        float(feature_rows[order[1]].get("cross_source_candidate_support_count", 0.0)) if len(order) > 1 else 0.0,
    ]


def entity_decision_feature_matrix(pair_df: pd.DataFrame,
                                   pair_probabilities: np.ndarray,
                                   entity_ids: Sequence[str],
                                   pair_threshold: float) -> np.ndarray:
    positions_by_entity: Dict[str, List[int]] = defaultdict(list)
    for position, entity_id in enumerate(pair_df["entity_id"].astype(str)):
        positions_by_entity[entity_id].append(position)
    rows: List[List[float]] = []
    all_features = pair_df["features"].tolist() if not pair_df.empty else []
    for entity_id in entity_ids:
        positions = positions_by_entity.get(str(entity_id), [])
        rows.append(entity_decision_feature_vector(
            [all_features[position] for position in positions],
            pair_probabilities[positions] if positions else np.asarray([], dtype=float),
            pair_threshold,
        ))
    return np.asarray(rows, dtype=float)


def scored_pair_frame(model: Any,
                      pair_df: pd.DataFrame,
                      feature_names: Sequence[str] = FEATURE_NAMES) -> Tuple[pd.DataFrame, np.ndarray]:
    if pair_df.empty:
        return pair_df.copy(), np.asarray([], dtype=float)
    if isinstance(model, NeuralReranker):
        top_pairs = select_top_k_pairs(pair_df, model.base_model, model.top_k)
        return top_pairs, predict_pair_probabilities(model.reranker, top_pairs)
    return pair_df, predict_pair_probabilities(model, pair_df, feature_names)


def predictions_from_pair_scores(pair_df: pd.DataFrame,
                                 probabilities: np.ndarray,
                                 threshold: float,
                                 score_margin: float | None,
                                 entity_ids: Sequence[str]) -> Dict[str, Set[str]]:
    predictions = {str(entity_id): set() for entity_id in entity_ids}
    if pair_df.empty:
        return predictions
    entity_positions = pair_df["entity_id"].astype(str).to_numpy()
    maximum_by_entity: Dict[str, float] = {}
    for entity_id, probability in zip(entity_positions, probabilities):
        maximum_by_entity[entity_id] = max(maximum_by_entity.get(entity_id, -np.inf), float(probability))
    for row, probability in zip(pair_df.itertuples(index=False), probabilities):
        entity_id = str(row.entity_id)
        if probability < threshold:
            continue
        if score_margin is not None and probability < maximum_by_entity[entity_id] - score_margin:
            continue
        predictions.setdefault(entity_id, set()).add(str(row.candidate_id))
    return predictions


def train_entity_decision_layer(model: Any,
                                val_df: pd.DataFrame,
                                truth: Mapping[str, Set[str]],
                                validation_entity_ids: Sequence[str],
                                pair_threshold: float,
                                score_margin: float | None,
                                seed: int) -> Tuple[EntityDecisionLayer | None, float, float]:
    entity_ids = [str(entity_id) for entity_id in validation_entity_ids]
    labels = np.asarray([bool(truth.get(entity_id, set())) for entity_id in entity_ids], dtype=int)
    counts = np.bincount(labels, minlength=2)
    scored_pairs, pair_probabilities = scored_pair_frame(model, val_df)
    pair_predictions = predictions_from_pair_scores(
        scored_pairs, pair_probabilities, pair_threshold, score_margin, entity_ids
    )
    # Reserve separate entity groups for gate fitting, threshold choice, and
    # the final gate-versus-pair-only comparison. Small samples skip the gate.
    if val_df.empty or len(entity_ids) < 160 or counts.min() < 80:
        baseline = score_entity_predictions(pair_predictions, truth, entity_ids)
        return None, baseline, baseline

    base_model = model.base_model if isinstance(model, NeuralReranker) else model
    all_base_probabilities = predict_pair_probabilities(base_model, val_df)
    entity_features = entity_decision_feature_matrix(
        val_df, all_base_probabilities, entity_ids, pair_threshold
    )
    all_indices = np.arange(len(entity_ids))
    fit_indices, holdout_indices = train_test_split(
        all_indices, test_size=0.5, random_state=seed, stratify=labels
    )
    tune_indices, eval_indices = train_test_split(
        holdout_indices, test_size=0.5, random_state=seed + 1,
        stratify=labels[holdout_indices],
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(entity_features[fit_indices], labels[fit_indices])

    tune_ids = [entity_ids[position] for position in tune_indices]
    gate_tune_probabilities = classifier.predict_proba(entity_features[tune_indices])[:, 1]
    thresholds = np.unique(np.concatenate((
        [0.0], np.quantile(gate_tune_probabilities, np.linspace(0.0, 1.0, 101)),
        [np.nextafter(float(gate_tune_probabilities.max()), np.inf)],
    )))
    best_threshold = float(thresholds[0])
    best_score = -1.0
    for threshold in thresholds:
        gated_predictions = {
            entity_id: pair_predictions.get(entity_id, set()) if probability >= threshold else set()
            for entity_id, probability in zip(tune_ids, gate_tune_probabilities)
        }
        score = score_entity_predictions(gated_predictions, truth, tune_ids)
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_score = score
            best_threshold = float(threshold)

    eval_ids = [entity_ids[position] for position in eval_indices]
    baseline_score = score_entity_predictions(pair_predictions, truth, eval_ids)
    gate_eval_probabilities = classifier.predict_proba(entity_features[eval_indices])[:, 1]
    gated_predictions = {
        entity_id: pair_predictions.get(entity_id, set()) if probability >= best_threshold else set()
        for entity_id, probability in zip(eval_ids, gate_eval_probabilities)
    }
    eval_score = score_entity_predictions(gated_predictions, truth, eval_ids)
    if eval_score <= baseline_score:
        return None, baseline_score, eval_score
    return EntityDecisionLayer(classifier=classifier, threshold=best_threshold), baseline_score, eval_score


def best_gate_threshold(gate_probabilities: np.ndarray,
                        pair_predictions: Mapping[str, Set[str]],
                        truth: Mapping[str, Set[str]],
                        entity_ids: Sequence[str]) -> float:
    ids = [str(entity_id) for entity_id in entity_ids]
    base_entity_scores = np.asarray([
        score_entity_predictions(pair_predictions, truth, [entity_id]) for entity_id in ids
    ], dtype=float)
    empty_entity_scores = np.asarray([
        1.0 if not truth.get(entity_id, set()) else 0.0 for entity_id in ids
    ], dtype=float)
    total_score = float(base_entity_scores.sum())
    best_score = total_score / max(len(ids), 1)
    best_threshold = 0.0
    order = np.argsort(gate_probabilities, kind="stable")
    start = 0
    while start < len(order):
        end = start + 1
        probability = float(gate_probabilities[order[start]])
        while end < len(order) and gate_probabilities[order[end]] == probability:
            end += 1
        positions = order[start:end]
        total_score += float(np.sum(empty_entity_scores[positions] - base_entity_scores[positions]))
        score = total_score / max(len(ids), 1)
        threshold = float(np.nextafter(probability, np.inf))
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_score = score
            best_threshold = threshold
        start = end
    return best_threshold


def fit_binary_score_calibrator(scores: np.ndarray,
                                labels: np.ndarray,
                                seed: int) -> Any | None:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if not len(scores) or len(np.unique(labels)) < 2:
        return None
    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    calibrator.fit(scores.reshape(-1, 1), labels)
    return calibrator


def apply_binary_score_calibrator(calibrator: Any | None, scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if calibrator is None or not len(scores):
        return scores
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def fit_cardinality_probability_calibrator(probabilities: np.ndarray,
                                           labels: np.ndarray,
                                           seed: int) -> Any | None:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(probabilities) == 0 or len(np.unique(labels)) < 2:
        return None
    logits = np.log(np.clip(probabilities, 1e-6, 1.0))
    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, random_state=seed),
    )
    calibrator.fit(logits, labels)
    return calibrator


def apply_cardinality_probability_calibrator(calibrator: Any | None,
                                             probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(probabilities):
        return np.empty((0, 3), dtype=float)
    fixed = np.zeros((len(probabilities), 3), dtype=float)
    fixed[:, :min(3, probabilities.shape[1])] = probabilities[:, :3]
    row_sums = fixed.sum(axis=1)
    fixed = np.divide(fixed, row_sums[:, None], out=np.zeros_like(fixed),
                      where=row_sums[:, None] > 0)
    if calibrator is None:
        return fixed
    calibrated = calibrator.predict_proba(np.log(np.clip(fixed, 1e-6, 1.0)))
    output = np.zeros_like(fixed)
    for source_position, label in enumerate(calibrator.classes_.astype(int)):
        if 0 <= label <= 2:
            output[:, label] = calibrated[:, source_position]
    row_sums = output.sum(axis=1)
    return np.divide(output, row_sums[:, None], out=fixed.copy(),
                     where=row_sums[:, None] > 0)


def cardinality_probability_matrix(classifier: Any,
                                   features: np.ndarray,
                                   probability_calibrator: Any | None = None) -> np.ndarray:
    raw = classifier.predict_proba(features)
    fixed = np.zeros((len(raw), 3), dtype=float)
    for source_position, label in enumerate(np.asarray(classifier.classes_, dtype=int)):
        if 0 <= label <= 2:
            fixed[:, label] = raw[:, source_position]
    return apply_cardinality_probability_calibrator(probability_calibrator, fixed)


def cardinality_labels_from_probability_matrix(
        probabilities: np.ndarray,
        confidence_thresholds: Mapping[int, float] | None = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Apply per-class confidence gates, allowing uncertain zero to defer to 1/2."""
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = {int(key): float(value)
                  for key, value in (confidence_thresholds or {}).items()}
    labels = np.zeros(len(probabilities), dtype=int)
    confidence = np.zeros(len(probabilities), dtype=float)
    for position, row in enumerate(probabilities):
        confidence[position] = float(np.max(row))
        raw_label = int(np.argmax(row))
        if row[raw_label] >= thresholds.get(raw_label, 0.0):
            labels[position] = raw_label
            continue
        eligible = [label for label in (1, 2)
                    if row[label] > 0.0 and row[label] >= thresholds.get(label, 0.0)]
        if eligible:
            selected = max(eligible, key=lambda label: (row[label], -label))
            labels[position] = selected
        else:
            labels[position] = 0
    return labels, confidence


def cardinality_labels_from_probabilities(
        classifier: Any,
        features: np.ndarray,
        confidence_threshold: float = 0.0,
        confidence_thresholds: Mapping[int, float] | None = None,
        probability_calibrator: Any | None = None,
        ) -> Tuple[np.ndarray, np.ndarray]:
    thresholds = (confidence_thresholds if confidence_thresholds is not None
                  else {0: confidence_threshold, 1: confidence_threshold,
                        2: confidence_threshold})
    probabilities = cardinality_probability_matrix(
        classifier, features, probability_calibrator
    )
    return cardinality_labels_from_probability_matrix(probabilities, thresholds)


def cardinality_adjusted_predictions(pair_df: pd.DataFrame,
                                     probabilities: np.ndarray,
                                     threshold: float,
                                     score_margin: float | None,
                                     entity_ids: Sequence[str],
                                     predicted_cardinality: np.ndarray | None,
                                     cardinality_thresholds: Mapping[int, float] | None = None,
                                     cardinality_margins: Mapping[int, float | None] | None = None,
                                     ) -> Dict[str, Set[str]]:
    predictions = predictions_from_pair_scores(
        pair_df, probabilities, threshold, score_margin, entity_ids
    )
    if predicted_cardinality is None:
        return predictions
    cardinality_thresholds = cardinality_thresholds or {}
    cardinality_margins = cardinality_margins or {}
    entity_to_cardinality = {
        str(entity_id): int(cardinality)
        for entity_id, cardinality in zip(entity_ids, predicted_cardinality)
    }
    positions_by_entity: Dict[str, List[int]] = defaultdict(list)
    for position, entity_id in enumerate(pair_df["entity_id"].astype(str)):
        positions_by_entity[entity_id].append(position)
    for entity_id in entity_ids:
        entity_id = str(entity_id)
        cardinality = entity_to_cardinality.get(entity_id, 0)
        positions = positions_by_entity.get(entity_id, [])
        if cardinality <= 0 or not positions:
            predictions[entity_id] = set()
            continue
        ranked = sorted(positions, key=lambda position: (-probabilities[position],
                                                        str(pair_df.iloc[position]["candidate_id"])))
        if cardinality == 1:
            best_position = ranked[0]
            predictions[entity_id] = (
                {str(pair_df.iloc[best_position]["candidate_id"])}
                if probabilities[best_position] >= cardinality_thresholds.get(1, threshold) else set()
            )
            continue
        class_threshold = cardinality_thresholds.get(2, threshold)
        class_margin = cardinality_margins.get(2, score_margin)
        best_probability = float(probabilities[ranked[0]])
        predictions[entity_id] = {
            str(pair_df.iloc[position]["candidate_id"])
            for position in ranked
            if probabilities[position] >= class_threshold
            and (class_margin is None or probabilities[position] >= best_probability - class_margin)
        }
    return predictions


def best_top1_threshold(pair_df: pd.DataFrame,
                        probabilities: np.ndarray,
                        truth: Mapping[str, Set[str]],
                        entity_ids: Sequence[str]) -> float:
    """Tune the singleton-cardinality class to choose only its best pair."""
    if pair_df.empty or not len(probabilities):
        return 1.000001
    positions_by_entity: Dict[str, List[int]] = defaultdict(list)
    for position, entity_id in enumerate(pair_df["entity_id"].astype(str)):
        positions_by_entity[entity_id].append(position)
    best_by_entity: Dict[str, Tuple[float, str]] = {}
    for entity_id, positions in positions_by_entity.items():
        best_position = min(
            positions,
            key=lambda position: (-float(probabilities[position]),
                                  str(pair_df.iloc[position]["candidate_id"])),
        )
        best_by_entity[entity_id] = (
            float(probabilities[best_position]),
            str(pair_df.iloc[best_position]["candidate_id"]),
        )
    boundaries = [0.0]
    boundaries.extend(float(np.nextafter(score, np.inf))
                       for score in sorted({value[0] for value in best_by_entity.values()}))
    best_threshold = 0.0
    best_score = -1.0
    for threshold in boundaries:
        predictions = {
            entity_id: ({best_by_entity[entity_id][1]} if entity_id in best_by_entity
                        and best_by_entity[entity_id][0] >= threshold else set())
            for entity_id in entity_ids
        }
        score = score_entity_predictions(predictions, truth, entity_ids)
        if score > best_score or (np.isclose(score, best_score) and threshold > best_threshold):
            best_score = score
            best_threshold = threshold
    return best_threshold


def tune_cardinality_policy(pair_df: pd.DataFrame,
                            probabilities: np.ndarray,
                            truth: Mapping[str, Set[str]],
                            entity_ids: Sequence[str],
                            predicted_cardinality: np.ndarray,
                            fallback_threshold: float,
                            fallback_margin: float | None) -> Tuple[Dict[int, float],
                                                                    Dict[int, float | None],
                                                                    float]:
    ids = [str(entity_id) for entity_id in entity_ids]
    classes = {entity_id: int(value)
               for entity_id, value in zip(ids, predicted_cardinality)}
    top1_ids = [entity_id for entity_id in ids if classes.get(entity_id, 0) == 1]
    multi_ids = [entity_id for entity_id in ids if classes.get(entity_id, 0) >= 2]
    thresholds: Dict[int, float] = {1: fallback_threshold, 2: fallback_threshold}
    margins: Dict[int, float | None] = {1: None, 2: fallback_margin}
    if top1_ids:
        top1_rows = pair_df[pair_df["entity_id"].astype(str).isin(set(top1_ids))]
        top1_probabilities = probabilities[
            pair_df["entity_id"].astype(str).isin(set(top1_ids)).to_numpy()
        ]
        thresholds[1] = best_top1_threshold(top1_rows, top1_probabilities, truth, top1_ids)
    if multi_ids:
        multi_mask = pair_df["entity_id"].astype(str).isin(set(multi_ids)).to_numpy()
        multi_rows = pair_df.loc[multi_mask]
        multi_probabilities = probabilities[multi_mask]
        thresholds[2], margins[2], _ = threshold_search_from_scores(
            multi_rows, multi_probabilities, truth, multi_ids
        )
    predictions = cardinality_adjusted_predictions(
        pair_df, probabilities, fallback_threshold, fallback_margin, ids,
        predicted_cardinality, thresholds, margins,
    )
    return thresholds, margins, score_entity_predictions(predictions, truth, ids)


def tune_joint_cardinality_policy(
        pair_df: pd.DataFrame,
        pair_probabilities: np.ndarray,
        truth: Mapping[str, Set[str]],
        entity_ids: Sequence[str],
        cardinality_probabilities: np.ndarray,
        fallback_threshold: float,
        fallback_margin: float | None,
        gate_probabilities: np.ndarray | None = None,
        gate_threshold: float = 0.0,
        ) -> Tuple[Dict[int, float], Dict[int, float],
                   Dict[int, float | None], float]:
    """Coordinate-tune P(0/1/2+) confidence and top-1/multi pair policy."""
    ids = [str(value) for value in entity_ids]
    probabilities = np.asarray(cardinality_probabilities, dtype=float)
    if probabilities.shape != (len(ids), 3):
        raise ValueError("Cardinality probabilities must have one P(0/1/2+) row per entity.")

    tune_rows = pair_df
    tune_pair_probabilities = np.asarray(pair_probabilities, dtype=float)
    if gate_probabilities is not None:
        gate_probabilities = np.asarray(gate_probabilities, dtype=float)
        allowed_ids = {entity_id for entity_id, score in zip(ids, gate_probabilities)
                       if score >= gate_threshold}
        row_mask = pair_df["entity_id"].astype(str).isin(allowed_ids).to_numpy()
        tune_rows = pair_df.loc[row_mask]
        tune_pair_probabilities = tune_pair_probabilities[row_mask]

    confidence_thresholds: Dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.0}

    def evaluate(thresholds: Mapping[int, float]
                 ) -> Tuple[Dict[int, float], Dict[int, float | None], float]:
        labels, _ = cardinality_labels_from_probability_matrix(probabilities, thresholds)
        pair_thresholds, pair_margins, _ = tune_cardinality_policy(
            tune_rows, tune_pair_probabilities, truth, ids, labels,
            fallback_threshold, fallback_margin,
        )
        predictions = cardinality_adjusted_predictions(
            tune_rows, tune_pair_probabilities, fallback_threshold, fallback_margin,
            ids, labels, pair_thresholds, pair_margins,
        )
        return pair_thresholds, pair_margins, score_entity_predictions(predictions, truth, ids)

    pair_thresholds, pair_margins, best_score = evaluate(confidence_thresholds)
    for _ in range(2):
        for class_label in (0, 1, 2):
            class_scores = probabilities[:, class_label]
            candidates = set(float(value) for value in np.quantile(
                class_scores, (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)
            ))
            candidates.add(float(np.nextafter(float(class_scores.max()), np.inf)))
            candidates.add(0.0)
            best_value = confidence_thresholds[class_label]
            best_class_thresholds = pair_thresholds
            best_class_margins = pair_margins
            for value in sorted(candidates):
                trial = dict(confidence_thresholds)
                trial[class_label] = value
                trial_pair_thresholds, trial_pair_margins, score = evaluate(trial)
                if score > best_score and not np.isclose(score, best_score):
                    best_score = score
                    best_value = value
                    best_class_thresholds = trial_pair_thresholds
                    best_class_margins = trial_pair_margins
            confidence_thresholds[class_label] = best_value
            pair_thresholds = best_class_thresholds
            pair_margins = best_class_margins
    return confidence_thresholds, pair_thresholds, pair_margins, best_score


def train_oof_entity_decision_layers(result: OOFModelResult,
                                     truth: Mapping[str, Set[str]],
                                     folds: int,
                                     threshold: float,
                                     score_margin: float | None) -> Tuple[EntityDecisionLayer | None,
                                                                          float, float, float]:
    """Tune on the last non-holdout fold, refit on folds 1..N-1, audit fold zero."""
    pair_rows = result.pair_rows
    entity_ids = result.entity_ids
    fold_by_entity = result.fold_by_entity
    if folds < 3 or not len(entity_ids):
        return None, 0.0, 0.0, 0.0

    tune_fold = folds - 1
    fit_mask = (fold_by_entity > 0) & (fold_by_entity < tune_fold)
    refit_mask = fold_by_entity > 0
    tune_mask = fold_by_entity == tune_fold
    eval_mask = fold_by_entity == 0
    linked_labels = np.asarray([bool(truth.get(entity_id, set())) for entity_id in entity_ids], dtype=int)
    cardinality_labels = np.asarray([min(2, len(truth.get(entity_id, set())))
                                     for entity_id in entity_ids], dtype=int)
    entity_features = result.entity_features.copy()
    entity_pos = {entity_id: position for position, entity_id in enumerate(entity_ids)}
    calibrated_all = (result.crossfit_probabilities if len(result.crossfit_probabilities) == len(pair_rows)
                      else apply_probability_calibrator(
                          result.probability_calibrator,
                          pair_rows["probability"].to_numpy(dtype=float),
                      ))
    pair_positions = pair_rows["entity_id"].astype(str).map(entity_pos).to_numpy(dtype=np.int64)
    above_threshold = np.bincount(
        pair_positions,
        weights=(calibrated_all >= threshold).astype(int),
        minlength=len(entity_ids),
    )
    entity_features[:, ENTITY_FEATURE_NAMES.index("number_above_pair_threshold")] = above_threshold
    result.entity_features = entity_features

    fit_positions = np.flatnonzero(fit_mask)
    refit_positions = np.flatnonzero(refit_mask)
    tune_positions = np.flatnonzero(tune_mask)
    eval_positions = np.flatnonzero(eval_mask)
    if len(fit_positions) < 20 or len(tune_positions) == 0 or len(eval_positions) == 0:
        return None, 0.0, 0.0, 0.0

    # Non-holdout pair decisions use their cross-fitted calibrated scores.
    # Fold 0 and test inference use the pooled calibrator; fold 0 remains an
    # independent audit of that deployment transform.
    pair_predictions_by_fold: Dict[int, Dict[str, Set[str]]] = {}
    for fold in range(folds):
        fold_rows = pair_rows[pair_rows["fold"].astype(int) == fold]
        fold_probabilities = (result.crossfit_probabilities[fold_rows.index.to_numpy(dtype=np.int64)]
                              if len(result.crossfit_probabilities) == len(pair_rows) else
                              apply_probability_calibrator(
                                  result.probability_calibrator,
                                  fold_rows["probability"].to_numpy(dtype=float),
                              ))
        fold_ids = [entity_id for position, entity_id in enumerate(entity_ids)
                    if fold_by_entity[position] == fold]
        pair_predictions_by_fold[fold] = predictions_from_pair_scores(
            fold_rows, fold_probabilities,
            threshold, score_margin, fold_ids,
        )

    fit_linked = linked_labels[fit_positions]
    gate_classifier = None
    gate_probability_calibrator = None
    gate_threshold = 1.1
    tune_gate_probabilities: np.ndarray | None = None
    tune_ids = [entity_ids[position] for position in tune_positions]
    eval_ids = [entity_ids[position] for position in eval_positions]
    tune_predictions = pair_predictions_by_fold[tune_fold]
    eval_predictions = pair_predictions_by_fold[0]
    tune_pair_score = score_entity_predictions(tune_predictions, truth, tune_ids)
    baseline_score = score_entity_predictions(eval_predictions, truth, eval_ids)
    if len(np.unique(fit_linked)) == 2 and np.bincount(fit_linked, minlength=2).min() >= 10:
        candidate_gate = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=0,
        ))
        candidate_gate.fit(entity_features[fit_positions], fit_linked)
        inner_gate_scores: List[float] = []
        inner_gate_labels: List[int] = []
        for inner_fold in range(1, tune_fold):
            inner_train_positions = fit_positions[fold_by_entity[fit_positions] != inner_fold]
            inner_eval_positions = np.flatnonzero(fold_by_entity == inner_fold)
            inner_labels = linked_labels[inner_train_positions]
            if (not len(inner_eval_positions) or len(np.unique(inner_labels)) < 2
                    or np.bincount(inner_labels, minlength=2).min() < 5):
                continue
            inner_gate = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=inner_fold,
            ))
            inner_gate.fit(entity_features[inner_train_positions], inner_labels)
            inner_gate_scores.extend(inner_gate.predict_proba(
                entity_features[inner_eval_positions]
            )[:, 1].tolist())
            inner_gate_labels.extend(linked_labels[inner_eval_positions].tolist())
        gate_probability_calibrator = fit_binary_score_calibrator(
            np.asarray(inner_gate_scores, dtype=float),
            np.asarray(inner_gate_labels, dtype=int), 200,
        )
        raw_tune_gate_probabilities = candidate_gate.predict_proba(
            entity_features[tune_positions]
        )[:, 1]
        tune_gate_probabilities = apply_binary_score_calibrator(
            gate_probability_calibrator, raw_tune_gate_probabilities
        )
        gate_threshold = best_gate_threshold(tune_gate_probabilities, tune_predictions, truth, tune_ids)
        tune_gated_predictions = {
            entity_id: tune_predictions.get(entity_id, set())
            if probability >= gate_threshold else set()
            for entity_id, probability in zip(tune_ids, tune_gate_probabilities)
        }
        gated_tune_score = score_entity_predictions(tune_gated_predictions, truth, tune_ids)
        if gated_tune_score > tune_pair_score:
            gate_classifier = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=0,
            ))
            gate_classifier.fit(entity_features[refit_positions], linked_labels[refit_positions])
            tune_pair_score = gated_tune_score
        else:
            gate_probability_calibrator = None

    holdout_rows = pair_rows[pair_rows["fold"].astype(int) == 0]
    holdout_probabilities = (result.crossfit_probabilities[holdout_rows.index.to_numpy(dtype=np.int64)]
                             if len(result.crossfit_probabilities) == len(pair_rows) else
                             apply_probability_calibrator(
                                 result.probability_calibrator,
                                 holdout_rows["probability"].to_numpy(dtype=float),
                             ))
    gate_score = baseline_score
    if gate_classifier is not None:
        eval_gate_probabilities = apply_binary_score_calibrator(
            gate_probability_calibrator,
            gate_classifier.predict_proba(entity_features[eval_positions])[:, 1],
        )
        gated_eval_predictions = {
            entity_id: eval_predictions.get(entity_id, set()) if probability >= gate_threshold else set()
            for entity_id, probability in zip(eval_ids, eval_gate_probabilities)
        }
        gate_score = score_entity_predictions(gated_eval_predictions, truth, eval_ids)

    cardinality_classifier = None
    cardinality_probability_calibrator = None
    cardinality_thresholds: Dict[int, float] = {}
    cardinality_margins: Dict[int, float | None] = {}
    cardinality_confidence_thresholds: Dict[int, float] = {}
    cardinality_score = gate_score
    fit_cardinality = cardinality_labels[fit_positions]
    class_counts = np.bincount(fit_cardinality, minlength=3)
    if np.count_nonzero(class_counts >= 10) >= 2:
        candidate_cardinality = make_pipeline(StandardScaler(), LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=1,
        ))
        candidate_cardinality.fit(entity_features[fit_positions], fit_cardinality)
        tune_pair_rows = pair_rows[pair_rows["fold"].astype(int) == tune_fold]
        tune_pair_probabilities = (
            result.crossfit_probabilities[tune_pair_rows.index.to_numpy(dtype=np.int64)]
            if len(result.crossfit_probabilities) == len(pair_rows) else
            apply_probability_calibrator(
                result.probability_calibrator,
                tune_pair_rows["probability"].to_numpy(dtype=float),
            )
        )
        inner_cardinality_probabilities: List[np.ndarray] = []
        inner_cardinality_labels: List[np.ndarray] = []
        for inner_fold in range(1, tune_fold):
            inner_train_positions = fit_positions[fold_by_entity[fit_positions] != inner_fold]
            inner_eval_positions = np.flatnonzero(fold_by_entity == inner_fold)
            inner_labels = cardinality_labels[inner_train_positions]
            if not len(inner_eval_positions) or len(np.unique(inner_labels)) < 2:
                continue
            inner_cardinality = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=inner_fold + 1,
            ))
            inner_cardinality.fit(entity_features[inner_train_positions], inner_labels)
            inner_cardinality_probabilities.append(cardinality_probability_matrix(
                inner_cardinality, entity_features[inner_eval_positions]
            ))
            inner_cardinality_labels.append(cardinality_labels[inner_eval_positions])
        if inner_cardinality_probabilities:
            cardinality_probability_calibrator = fit_cardinality_probability_calibrator(
                np.vstack(inner_cardinality_probabilities),
                np.concatenate(inner_cardinality_labels), 201,
            )
        tune_cardinality_probabilities = cardinality_probability_matrix(
            candidate_cardinality, entity_features[tune_positions],
            cardinality_probability_calibrator,
        )
        candidate_gate_for_tune = (tune_gate_probabilities
                                   if gate_classifier is not None else None)
        (candidate_confidence_thresholds, candidate_thresholds, candidate_margins,
         tune_cardinality_score) = tune_joint_cardinality_policy(
            tune_pair_rows, tune_pair_probabilities, truth, tune_ids,
            tune_cardinality_probabilities, threshold, score_margin,
            candidate_gate_for_tune, gate_threshold,
        )
        if tune_cardinality_score > tune_pair_score:
            cardinality_classifier = make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=1,
            ))
            cardinality_classifier.fit(entity_features[refit_positions],
                                       cardinality_labels[refit_positions])
            cardinality_thresholds = candidate_thresholds
            cardinality_margins = candidate_margins
            cardinality_confidence_thresholds = candidate_confidence_thresholds
            tune_pair_score = tune_cardinality_score
            eval_cardinality_probabilities = cardinality_probability_matrix(
                cardinality_classifier, entity_features[eval_positions],
                cardinality_probability_calibrator,
            )
            eval_cardinality, _ = cardinality_labels_from_probability_matrix(
                eval_cardinality_probabilities, cardinality_confidence_thresholds,
            )
            cardinality_predictions = cardinality_adjusted_predictions(
                holdout_rows, holdout_probabilities, threshold, score_margin, eval_ids,
                eval_cardinality, cardinality_thresholds, cardinality_margins,
            )
            if gate_classifier is not None:
                eval_gate_probabilities = apply_binary_score_calibrator(
                    gate_probability_calibrator,
                    gate_classifier.predict_proba(entity_features[eval_positions])[:, 1],
                )
                cardinality_predictions = {
                    entity_id: cardinality_predictions.get(entity_id, set())
                    if probability >= gate_threshold else set()
                    for entity_id, probability in zip(eval_ids, eval_gate_probabilities)
                }
            cardinality_score = score_entity_predictions(cardinality_predictions, truth, eval_ids)

    if gate_classifier is None and cardinality_classifier is None:
        return None, baseline_score, gate_score, cardinality_score
    return EntityDecisionLayer(
        classifier=gate_classifier,
        threshold=gate_threshold,
        probability_calibrator=gate_probability_calibrator,
        cardinality_classifier=cardinality_classifier,
        cardinality_probability_calibrator=cardinality_probability_calibrator,
        cardinality_thresholds=cardinality_thresholds,
        cardinality_margins=cardinality_margins,
        cardinality_confidence_thresholds=cardinality_confidence_thresholds,
        feature_pair_threshold=threshold,
    ), baseline_score, gate_score, cardinality_score


def feature_matrix(pair_df: pd.DataFrame,
                   feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    if pair_df.empty:
        return np.empty((0, len(feature_names)), dtype=float)
    _require_candidate_metadata_enriched(pair_df["features"].tolist())
    return np.asarray([features_to_vector(value, feature_names) for value in pair_df["features"]], dtype=float)


def select_top_k_pairs(pair_df: pd.DataFrame,
                       scorer: Any,
                       top_k: int) -> pd.DataFrame:
    """Keep each entity's top-K pairs according to a first-stage model."""
    if pair_df.empty:
        return pair_df.copy()
    positions_by_entity: Dict[str, List[int]] = defaultdict(list)
    for position, entity_id in enumerate(pair_df["entity_id"].astype(str)):
        positions_by_entity[entity_id].append(position)
    selected_positions: List[int] = []
    for entity_id in sorted(positions_by_entity):
        positions = np.asarray(positions_by_entity[entity_id], dtype=np.int64)
        subset = pair_df.iloc[positions]
        scores = predict_pair_probabilities(scorer, subset)
        ordered = sorted(
            range(len(positions)),
            key=lambda offset: (-scores[offset], str(subset.iloc[offset]["candidate_id"])),
        )
        selected_positions.extend(positions[ordered[:top_k]].tolist())
    return pair_df.iloc[selected_positions].reset_index(drop=True)


def train_neural_reranker(train_df: pd.DataFrame,
                          val_df: pd.DataFrame,
                          base_model: Any,
                          top_k: int,
                          truth: Mapping[str, Set[str]],
                          validation_entity_ids: Sequence[str],
                          seed: int) -> Tuple[NeuralReranker, float, float | None, float]:
    reranker = fit_neural_reranker(train_df, base_model, top_k, seed)
    val_top = select_top_k_pairs(val_df, base_model, top_k)
    threshold, score_margin, validation_score = threshold_search(
        reranker, val_top, truth, validation_entity_ids
    )
    return NeuralReranker(base_model=base_model, reranker=reranker, top_k=top_k), threshold, score_margin, validation_score


def fit_neural_reranker(train_df: pd.DataFrame,
                        base_model: Any,
                        top_k: int,
                        seed: int) -> Any:
    train_top = select_top_k_pairs(train_df, base_model, top_k)
    if train_top.empty or train_top["label"].nunique() < 2:
        raise RuntimeError(
            "The top-K reranker training sample needs positive and negative candidate pairs; "
            "increase --neural-top-k or use more training entities."
        )
    labels = train_top["label"].astype(int).to_numpy()
    class_counts = np.bincount(labels, minlength=2)
    use_early_stopping = bool(len(train_top) >= 100 and class_counts.min() >= 10)
    classifier = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=use_early_stopping,
            n_iter_no_change=12,
            random_state=seed,
        ),
    )
    classifier.fit(feature_matrix(train_top), labels)
    return classifier


def train_model(train_df: pd.DataFrame,
                val_df: pd.DataFrame,
                truth: Mapping[str, Set[str]],
                validation_entity_ids: Sequence[str],
                seed: int = 42,
                model_type: str = "logistic",
                feature_names: Sequence[str] = FEATURE_NAMES) -> Tuple[Any, float, float | None, float]:
    if model_type == "compare":
        if LGBMClassifier is None:
            raise RuntimeError("Install requirements.txt to compare logistic regression with LightGBM.")
        results = []
        for candidate_model_type in ("logistic", "lightgbm"):
            result = train_model(train_df, val_df, truth, validation_entity_ids,
                                 seed=seed, model_type=candidate_model_type,
                                 feature_names=feature_names)
            print(f"Validation entity-macro F0.5 — {candidate_model_type}: {result[3]:.4f}")
            results.append(result)
        return max(results, key=lambda result: result[3])

    if train_df.empty:
        raise RuntimeError("No candidate pairs were generated for training.")
    X_train = feature_matrix(train_df, feature_names)
    y_train = train_df["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training candidates contain only one class; improve blocking or sampling.")

    model = make_pair_estimator(model_type, seed)
    model.fit(X_train, y_train)
    threshold, score_margin, validation_score = threshold_search(
        model, val_df, truth, validation_entity_ids, feature_names
    )
    return model, threshold, score_margin, validation_score


def make_base_pair_estimator(model_type: str, seed: int) -> Any:
    model_family = model_type
    unweighted = model_family.endswith("_unweighted")
    if unweighted:
        model_family = model_family[:-len("_unweighted")]
    class_weight = None if unweighted else "balanced"
    if model_family == "logistic":
        return LogisticRegression(max_iter=3000, class_weight=class_weight, random_state=seed)
    if model_family == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is not installed. Install requirements.txt to use --model lightgbm.")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {model_type}")


def make_pair_estimator(model_type: str, seed: int) -> Any:
    if model_type.startswith("ensemble:"):
        member_types = [member.strip() for member in model_type.split(":", 1)[1].split(",")
                        if member.strip()]
        if len(member_types) < 2:
            raise ValueError("An ensemble needs at least two comma-separated model families.")
        return EnsemblePairEstimator([
            make_pair_estimator(member_type, seed) for member_type in member_types
        ])
    source_specific = model_type.endswith("_source_specific")
    base_type = (model_type[:-len("_source_specific")]
                 if source_specific else model_type)
    model = make_base_pair_estimator(base_type, seed)
    if not source_specific:
        return model
    return SourceSpecificPairEstimator(
        shared_model=model,
        source_feature_index=FEATURE_NAMES.index("candidate_source_s2"),
        base_model_type=base_type,
        seed=seed,
    )


def run_entity_oof_validation(source1: pd.DataFrame,
                              truth: Mapping[str, Set[str]],
                              source_lookup: pd.DataFrame,
                              index: BlockIndex,
                              model_types: Sequence[str],
                              folds: int,
                              seed: int,
                              negatives_per_positive: int,
                              random_negatives: int,
                              neural_top_k: int = 20,
                              adversarial_negatives_per_entity: int = 5,
                              ) -> Tuple[Dict[str, OOFModelResult], pd.DataFrame, float,
                                         CandidateDiagnostics]:
    entity_ids = source1["entity_id"].astype(str).tolist()
    linked_labels = np.asarray([bool(truth.get(entity_id, set())) for entity_id in entity_ids], dtype=int)
    class_counts = np.bincount(linked_labels, minlength=2)
    if folds < 3:
        raise ValueError("OOF validation needs at least 3 entity folds.")
    if class_counts.min() < folds:
        raise ValueError(
            f"OOF validation needs at least {folds} linked and singleton entities; "
            f"found {class_counts.tolist()}. Reduce --cv-folds or provide more entities."
        )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_by_entity = np.full(len(source1), -1, dtype=np.int64)
    fold_splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold, (train_positions, validation_positions) in enumerate(
        splitter.split(np.zeros(len(source1)), linked_labels)
    ):
        fold_by_entity[validation_positions] = fold
        fold_splits.append((train_positions, validation_positions))

    all_training_examples, train_candidate_recall, train_diagnostics = build_training_examples(
        source1, None, truth,
        negatives_per_positive=negatives_per_positive,
        random_negatives=random_negatives,
        training=True,
        seed=seed,
        source_lookup=source_lookup,
        index=index,
    )
    print(f"OOF training examples: {len(all_training_examples)}")
    print_candidate_diagnostics("OOF training pool", train_diagnostics, train_candidate_recall)

    oof_rows: Dict[str, List[pd.DataFrame]] = {model_type: [] for model_type in model_types}
    oof_entity_features = {
        model_type: np.zeros((len(source1), len(ENTITY_FEATURE_NAMES)), dtype=float)
        for model_type in model_types
    }
    mined_rows: Dict[str, List[pd.DataFrame]] = {model_type: [] for model_type in model_types}
    entity_position = {entity_id: position for position, entity_id in enumerate(entity_ids)}
    holdout_examples = pd.DataFrame()
    holdout_source1 = pd.DataFrame()
    holdout_base_probabilities: Dict[str, np.ndarray] = {}
    candidate_recall_by_fold: Dict[int, float] = {}
    topk_candidate_recall_by_model: Dict[str, Dict[int, float]] = {
        model_type: {} for model_type in model_types
    }
    model_runtime_seconds = {model_type: 0.0 for model_type in model_types}

    for fold, (train_positions, validation_positions) in enumerate(fold_splits):
        validation_source1 = source1.iloc[validation_positions]
        validation_ids = validation_source1["entity_id"].astype(str).tolist()
        train_ids = {entity_ids[position] for position in train_positions}
        fold_training_examples = all_training_examples[
            all_training_examples["entity_id"].astype(str).isin(train_ids)
        ]
        validation_examples, candidate_recall_value, diagnostics = build_training_examples(
            validation_source1, None, truth,
            training=False,
            seed=seed,
            source_lookup=source_lookup,
            index=index,
        )
        candidate_recall_by_fold[fold] = candidate_recall_for_rows(
            validation_examples, truth, validation_ids
        )
        if fold == 0:
            holdout_examples = validation_examples
            holdout_source1 = validation_source1.copy()
        print_candidate_diagnostics(f"OOF fold {fold + 1}/{folds}", diagnostics, candidate_recall_value)
        y_train = fold_training_examples["label"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"OOF fold {fold + 1} training pairs contain only one class.")
        for model_type in model_types:
            model_started = time.perf_counter()
            use_mlp = model_type.endswith("_mlp")
            base_type = model_type[:-4] if use_mlp else model_type
            model = make_pair_estimator(base_type, seed + fold)
            model.fit(feature_matrix(fold_training_examples), y_train)
            fold_mined = pd.DataFrame(columns=["entity_id", "candidate_id", "label", "features"])
            fit_training_examples = fold_training_examples
            if adversarial_negatives_per_entity > 0:
                mining_model: Any = model
                if use_mlp:
                    mining_model = NeuralReranker(
                        base_model=model,
                        reranker=fit_neural_reranker(
                            fold_training_examples, model, neural_top_k, seed + fold
                        ),
                        top_k=neural_top_k,
                    )
                fold_mined = mine_final_adversarial_negatives(
                    source1.iloc[train_positions], truth, source_lookup, index,
                    mining_model, fold_training_examples,
                    adversarial_negatives_per_entity, seed=seed + fold,
                )
                if not fold_mined.empty:
                    fit_training_examples = pd.concat(
                        [fold_training_examples, fold_mined], ignore_index=True
                    ).drop_duplicates(["entity_id", "candidate_id"], keep="first")
                    model = make_pair_estimator(base_type, seed + fold)
                    model.fit(feature_matrix(fit_training_examples),
                              fit_training_examples["label"].astype(int).to_numpy())
                    mined_rows[model_type].append(fold_mined)
            base_probabilities = predict_pair_probabilities(model, validation_examples)
            if fold == 0:
                holdout_base_probabilities[model_type] = base_probabilities.copy()
            scored_validation_examples = validation_examples
            scoring_model = model
            if use_mlp:
                scoring_model = fit_neural_reranker(
                    fit_training_examples, model, neural_top_k, seed + fold
                )
                scored_validation_examples = select_top_k_pairs(
                    validation_examples, model, neural_top_k
                )
            topk_candidate_recall_by_model[model_type][fold] = candidate_recall_for_rows(
                scored_validation_examples, truth, validation_ids
            )
            probabilities = predict_pair_probabilities(scoring_model, scored_validation_examples)
            fold_rows = scored_validation_examples[["entity_id", "candidate_id", "label"]].copy()
            fold_rows["probability"] = probabilities
            fold_rows["fold"] = fold
            oof_rows[model_type].append(fold_rows)

            gate_features = entity_decision_feature_matrix(
                scored_validation_examples, probabilities, validation_ids, pair_threshold=0.0
            )
            positions = [entity_position[entity_id] for entity_id in validation_ids]
            oof_entity_features[model_type][positions] = gate_features

            model_runtime_seconds[model_type] += time.perf_counter() - model_started
            del model

        del y_train, fold_training_examples, validation_examples
        gc.collect()

    results: Dict[str, OOFModelResult] = {}
    for model_type in model_types:
        pair_rows = pd.concat(oof_rows[model_type], ignore_index=True)
        mined = (pd.concat(mined_rows[model_type], ignore_index=True)
                 if mined_rows[model_type] else
                 pd.DataFrame(columns=["entity_id", "candidate_id", "label", "features"]))
        results[model_type] = OOFModelResult(
            model_type=model_type,
            pair_rows=pair_rows,
            entity_features=oof_entity_features[model_type],
            entity_ids=entity_ids,
            fold_by_entity=fold_by_entity.copy(),
            mined_negatives=mined,
            holdout_examples=holdout_examples,
            holdout_source1=holdout_source1,
            holdout_base_probabilities=holdout_base_probabilities.get(
                model_type, np.asarray([], dtype=float)
            ),
            candidate_recall_by_fold=dict(candidate_recall_by_fold),
            topk_candidate_recall_by_fold=dict(topk_candidate_recall_by_model[model_type]),
            model_runtime_seconds=model_runtime_seconds[model_type],
        )
    return results, all_training_examples, train_candidate_recall, train_diagnostics


def fit_probability_calibrator(pair_rows: pd.DataFrame, seed: int) -> Any | None:
    """Fit a compact Platt calibrator from out-of-fold pair scores."""
    if pair_rows.empty or pair_rows["label"].nunique() < 2:
        return None
    calibration_rows = pair_rows
    if len(calibration_rows) > 500_000:
        sampled_classes: List[pd.DataFrame] = []
        scale = 500_000 / len(calibration_rows)
        for label in (0, 1):
            class_rows = calibration_rows[calibration_rows["label"].astype(int) == label]
            sample_count = min(len(class_rows), max(1, int(round(len(class_rows) * scale))))
            sampled_classes.append(class_rows.sample(n=sample_count, random_state=seed + label))
        calibration_rows = pd.concat(sampled_classes, ignore_index=True)
    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=seed),
    )
    calibrator.fit(
        calibration_rows["probability"].to_numpy(dtype=float).reshape(-1, 1),
        calibration_rows["label"].astype(int).to_numpy(),
    )
    return calibrator


def apply_probability_calibrator(calibrator: Any | None,
                                 probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None or not len(probabilities):
        return np.asarray(probabilities, dtype=float)
    return calibrator.predict_proba(
        np.asarray(probabilities, dtype=float).reshape(-1, 1)
    )[:, 1]


def calibrate_and_compare_oof_models(results: Mapping[str, OOFModelResult],
                                     truth: Mapping[str, Set[str]],
                                     folds: int) -> OOFModelResult:
    for result in results.values():
        pair_rows = result.pair_rows
        fold_values = pair_rows["fold"].to_numpy(dtype=int)
        crossfit_probabilities = np.empty(len(pair_rows), dtype=float)
        fold_scores: List[float] = []
        result.cross_validation_fold_scores = {}
        result.match_recall_by_fold = {}
        # Fold 0 remains an independent final check. Every non-holdout fold is
        # calibrated by a calibrator that excluded that fold, so threshold
        # selection stays cross-fitted instead of reusing pooled-calibrator fit
        # scores. The pooled calibrator is reserved for fold 0 and deployment.
        for evaluation_fold in range(1, folds):
            calibration_mask = (fold_values != 0) & (fold_values != evaluation_fold)
            evaluation_mask = fold_values == evaluation_fold
            calibration_rows = pair_rows.loc[calibration_mask]
            evaluation_rows = pair_rows.loc[evaluation_mask]
            fold_calibrator = fit_probability_calibrator(calibration_rows, 100 + evaluation_fold)
            calibration_probabilities = apply_probability_calibrator(
                fold_calibrator, calibration_rows["probability"].to_numpy(dtype=float)
            )
            evaluation_probabilities = apply_probability_calibrator(
                fold_calibrator, evaluation_rows["probability"].to_numpy(dtype=float)
            )
            crossfit_probabilities[evaluation_rows.index.to_numpy(dtype=np.int64)] = evaluation_probabilities
            calibration_ids = [
                entity_id for position, entity_id in enumerate(result.entity_ids)
                if result.fold_by_entity[position] not in {0, evaluation_fold}
            ]
            evaluation_ids = [
                entity_id for position, entity_id in enumerate(result.entity_ids)
                if result.fold_by_entity[position] == evaluation_fold
            ]
            threshold, margin, _ = threshold_search_from_scores(
                calibration_rows, calibration_probabilities,
                truth, calibration_ids,
            )
            candidate_recall_value = result.candidate_recall_by_fold.get(
                evaluation_fold,
                candidate_recall_for_rows(evaluation_rows, truth, evaluation_ids),
            )
            metrics = evaluate_scored_pairs(
                evaluation_rows, evaluation_probabilities,
                truth, evaluation_ids, threshold, margin, candidate_recall_value,
            )
            fold_scores.append(metrics["entity_f0_5"])
            result.cross_validation_fold_scores[evaluation_fold] = float(metrics["entity_f0_5"])
            result.match_recall_by_fold[evaluation_fold] = metrics["pair_recall"]
        result.cross_validation_f0_5 = float(np.mean(fold_scores)) if fold_scores else 0.0
        fold_score_min = min(fold_scores, default=0.0)
        fold_score_max = max(fold_scores, default=0.0)
        fold_score_std = float(np.std(fold_scores)) if fold_scores else 0.0

        calibration_mask = fold_values != 0
        calibration_rows = pair_rows.loc[calibration_mask]
        result.probability_calibrator = fit_probability_calibrator(calibration_rows, 42)
        calibration_probabilities = crossfit_probabilities[
            calibration_rows.index.to_numpy(dtype=np.int64)
        ]
        calibration_ids = [
            entity_id for position, entity_id in enumerate(result.entity_ids)
            if result.fold_by_entity[position] != 0
        ]
        result.threshold, result.score_margin, _ = threshold_search_from_scores(
            calibration_rows, calibration_probabilities,
            truth, calibration_ids,
        )
        holdout_mask = fold_values == 0
        holdout_rows = pair_rows.loc[holdout_mask]
        holdout_probabilities = apply_probability_calibrator(
            result.probability_calibrator,
            holdout_rows["probability"].to_numpy(dtype=float),
        )
        crossfit_probabilities[holdout_rows.index.to_numpy(dtype=np.int64)] = holdout_probabilities
        result.crossfit_probabilities = crossfit_probabilities
        holdout_ids = [
            entity_id for position, entity_id in enumerate(result.entity_ids)
            if result.fold_by_entity[position] == 0
        ]
        result.heldout_metrics = evaluate_scored_pairs(
            holdout_rows, holdout_probabilities,
            truth, holdout_ids, result.threshold, result.score_margin,
            result.candidate_recall_by_fold.get(
                0, candidate_recall_for_rows(holdout_rows, truth, holdout_ids)
            ),
        )
        result.heldout_metrics["blocking_candidate_recall"] = result.candidate_recall_by_fold.get(
            0, candidate_recall_for_rows(holdout_rows, truth, holdout_ids)
        )
        result.heldout_metrics["base_model_topk_recall"] = (
            result.topk_candidate_recall_by_fold.get(
                0, result.heldout_metrics["blocking_candidate_recall"]
            )
        )
        result.heldout_metrics["reranker_match_recall"] = result.heldout_metrics["pair_recall"]
        print(f"OOF {result.model_type}: CV entity F0.5={result.cross_validation_f0_5:.4f} "
              f"(fold min/max/std={fold_score_min:.4f}/{fold_score_max:.4f}/{fold_score_std:.4f}); "
              f"reserved independent fold entity F0.5={result.heldout_metrics['entity_f0_5']:.4f}; "
              f"blocking recall={result.heldout_metrics['blocking_candidate_recall']:.4f}; "
              f"top-K recall={result.heldout_metrics['base_model_topk_recall']:.4f}; "
              f"matched recall={result.heldout_metrics['reranker_match_recall']:.4f}; "
              f"pair Brier={result.heldout_metrics['pair_brier_score']:.4f}; "
              f"pair log loss={result.heldout_metrics['pair_log_loss']:.4f}")
    return max(results.values(), key=lambda result: result.cross_validation_f0_5)


def write_oof_model_comparison_report(path: Path,
                                      results: Mapping[str, OOFModelResult],
                                      selected_model_type: str) -> pd.DataFrame:
    """Rank the evaluated matcher variants by non-holdout entity F0.5."""
    metric_names = (
        "entity_f0_5", "singleton_f0_5", "multi_match_f0_5", "source2_f0_5",
        "source3_f0_5", "candidate_pair_recall", "blocking_candidate_recall",
        "base_model_topk_recall", "reranker_match_recall", "pair_precision", "pair_recall",
        "pair_brier_score", "pair_log_loss",
    )
    rows: List[Dict[str, object]] = []
    for result in results.values():
        row: Dict[str, object] = {
            "model_type": result.model_type,
            "oof_cv_entity_f0_5": result.cross_validation_f0_5,
            "oof_cv_f0_5_min": min(result.cross_validation_fold_scores.values(), default=0.0),
            "oof_cv_f0_5_max": max(result.cross_validation_fold_scores.values(), default=0.0),
            "oof_cv_f0_5_std": (float(np.std(list(result.cross_validation_fold_scores.values())))
                                if result.cross_validation_fold_scores else 0.0),
            "oof_blocking_candidate_recall": float(np.mean([
                value for fold, value in result.candidate_recall_by_fold.items() if fold != 0
            ])) if any(fold != 0 for fold in result.candidate_recall_by_fold) else 0.0,
            "oof_base_model_topk_recall": float(np.mean([
                value for fold, value in result.topk_candidate_recall_by_fold.items() if fold != 0
            ])) if any(fold != 0 for fold in result.topk_candidate_recall_by_fold) else 0.0,
            "oof_match_recall": float(np.mean([
                value for fold, value in result.match_recall_by_fold.items() if fold != 0
            ])) if any(fold != 0 for fold in result.match_recall_by_fold) else 0.0,
            "oof_model_runtime_seconds": result.model_runtime_seconds,
            "selected_by_oof_cv": result.model_type == selected_model_type,
        }
        row.update({f"fold0_{name}": result.heldout_metrics.get(name, 0.0)
                    for name in metric_names})
        rows.append(row)
    report = pd.DataFrame(rows).sort_values(
        ["oof_cv_entity_f0_5", "oof_model_runtime_seconds", "model_type"],
        ascending=[False, True, True], kind="stable",
    ).reset_index(drop=True)
    report.insert(0, "oof_cv_rank", np.arange(1, len(report) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return report


def fit_final_pair_model(train_df: pd.DataFrame,
                         model_type: str,
                         seed: int,
                         neural_top_k: int) -> Any:
    use_mlp = model_type.endswith("_mlp")
    base_type = model_type[:-4] if use_mlp else model_type
    X_train = feature_matrix(train_df)
    y_train = train_df["label"].astype(int).to_numpy()
    if train_df.empty or len(np.unique(y_train)) < 2:
        raise RuntimeError("Final training examples need both positive and negative pairs.")
    base_model = make_pair_estimator(base_type, seed)
    base_model.fit(X_train, y_train)
    if not use_mlp:
        return base_model
    reranker = fit_neural_reranker(train_df, base_model, neural_top_k, seed)
    return NeuralReranker(base_model=base_model, reranker=reranker, top_k=neural_top_k)


def mine_final_adversarial_negatives(source1: pd.DataFrame,
                                     truth: Mapping[str, Set[str]],
                                     source_lookup: pd.DataFrame,
                                     index: BlockIndex,
                                     model: Any,
                                     existing_examples: pd.DataFrame,
                                     per_entity: int,
                                     seed: int = 42) -> pd.DataFrame:
    """Mine unsampled false candidates using the current full-data model."""
    columns = ["entity_id", "candidate_id", "label", "features"]
    if per_entity <= 0:
        return pd.DataFrame(columns=columns)
    seen_by_entity: Dict[str, Set[str]] = defaultdict(set)
    for row in existing_examples[["entity_id", "candidate_id"]].itertuples(index=False):
        seen_by_entity[str(row.entity_id)].add(str(row.candidate_id))
    mined_rows: List[Dict[str, object]] = []
    for row in source1.itertuples(index=False):
        s1_values = row._asdict()
        entity_id = str(s1_values.get("entity_id", ""))
        candidates, block_candidates = generate_candidate_details(
            s1_values, index, source_lookup, collect_blocks=True
        )
        true_ids = truth.get(entity_id, set())
        already_seen = seen_by_entity.get(entity_id, set())
        available = [candidate for candidate in candidates
                     if candidate not in true_ids and candidate not in already_seen]
        if not available:
            continue
        candidate_ranks = {candidate: position + 1
                           for position, candidate in enumerate(candidates)}
        metadata: List[Dict[str, float]] = [dict() for _ in candidates]
        add_candidate_block_memberships(metadata, candidates, block_candidates, index)
        candidate_position = {candidate: position for position, candidate in enumerate(candidates)}
        feature_rows: List[Dict[str, float]] = []
        for candidate_id in available:
            features = build_pair_features(
                s1_values, lookup_record(source_lookup, candidate_id), index,
                candidate_rank=candidate_ranks[candidate_id],
            )
            features.update(metadata[candidate_position[candidate_id]])
            feature_rows.append(features)
        if isinstance(model, NeuralReranker):
            base_probabilities = predict_feature_rows(model.base_model, feature_rows)
            top_positions = sorted(
                range(len(available)),
                key=lambda position: (-float(base_probabilities[position]), available[position]),
            )[:model.top_k]
            reranked_probabilities = predict_feature_rows(
                model.reranker, [feature_rows[position] for position in top_positions]
            )
            probabilities = np.full(len(available), -np.inf, dtype=float)
            probabilities[top_positions] = reranked_probabilities
        else:
            probabilities = predict_feature_rows(model, feature_rows)
        ranked = [
            (float(probabilities[position]), available[position], negative_category(feature_rows[position]))
            for position in range(len(available))
        ]
        stable_seed = int(hashlib.sha1(
            f"{seed}:{entity_id}".encode("utf-8")
        ).hexdigest()[:8], 16)
        chosen_ids = set(select_mixed_hard_negatives(
            ranked, per_entity, np.random.default_rng(stable_seed)
        ))
        chosen_positions = [position for position, candidate_id in enumerate(available)
                            if candidate_id in chosen_ids]
        for position in chosen_positions:
            mined_rows.append({
                "entity_id": entity_id,
                "candidate_id": available[position],
                "label": 0,
                "features": feature_rows[position],
            })
    return pd.DataFrame(mined_rows, columns=columns)


def score_distribution_row(label: str,
                           score_type: str,
                           scores: np.ndarray) -> Dict[str, object] | None:
    values = np.asarray(scores, dtype=float)
    if not len(values):
        return None
    return {
        "dataset": label,
        "score_type": score_type,
        "sample_count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p01": float(np.quantile(values, 0.01)),
        "p05": float(np.quantile(values, 0.05)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def score_distribution_rows(label: str,
                            probabilities: np.ndarray,
                            calibrator: Any | None) -> List[Dict[str, object]]:
    """Summarize raw and final-calibrator score distributions for drift review."""
    rows = [score_distribution_row(label, "raw", probabilities)]
    rows.append(score_distribution_row(
        label, "final_oof_calibrator", apply_probability_calibrator(calibrator, probabilities)
    ))
    return [row for row in rows if row is not None]


def write_calibration_drift_report(path: Path,
                                   oof_probabilities: np.ndarray,
                                   oof_threshold_probabilities: np.ndarray,
                                   final_model: Any,
                                   final_training_examples: pd.DataFrame,
                                   calibrator: Any | None,
                                   seed: int,
                                   sample_limit: int = 200_000) -> None:
    """Compare fold-0 pooled scores, cross-fitted threshold scores, and final scores."""
    rows = score_distribution_rows("oof_fold0_heldout", oof_probabilities, calibrator)
    crossfit_row = score_distribution_row(
        "oof_threshold_selection_nonholdout", "crossfit_calibrated",
        oof_threshold_probabilities,
    )
    if crossfit_row is not None:
        rows.append(crossfit_row)
    if not final_training_examples.empty:
        sample_count = min(len(final_training_examples), sample_limit)
        sample = final_training_examples
        if len(sample) > sample_count:
            sample = sample.sample(n=sample_count, random_state=seed)
        _, final_scores = scored_pair_frame(final_model, sample)
        rows.extend(score_distribution_rows("final_training_in_sample", final_scores, calibrator))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def candidate_recall_for_rows(pair_rows: pd.DataFrame,
                              truth: Mapping[str, Set[str]],
                              entity_ids: Sequence[str]) -> float:
    total_truth = sum(len(truth.get(str(entity_id), set())) for entity_id in entity_ids)
    found_truth = int(pair_rows["label"].astype(int).sum()) if not pair_rows.empty else 0
    return found_truth / total_truth if total_truth else 1.0


def feature_ablation_sets() -> Dict[str, List[str]]:
    address_prefixes = ("address_", "common_address", "number_", "house_number", "unit_number",
                        "postal_code", "city_", "state_", "tail_component", "phone_fragment",
                        "address_landmark", "postal_country_pattern_support",
                        "candidate_retrieval_score")
    location_prefixes = ("number_", "house_number", "unit_number", "postal_code", "city_",
                         "state_", "tail_component", "same_country", "country_conflict",
                         "phone_fragment", "address_landmark", "address_layout_",
                         "postal_country_pattern_support", "candidate_retrieval_score")
    character_features = {
        "name_char_similarity", "name_jaro_winkler", "name_char_ngram_jaccard",
        "name_char_ngram_cosine", "address_char_similarity", "address_char_ngram_jaccard",
        "address_char_ngram_cosine", "name_sorted_token_similarity",
    }
    tfidf_features = {
        "name_tfidf_cosine", "name_rare_token_overlap", "address_tfidf_cosine",
        "address_rare_token_overlap",
    }
    cross_field_features = {
        "name_address_consistency", "city_name_consistency", "name_postal_consistency",
    }
    cross_source_features = {
        name for name in FEATURE_NAMES
        if name.startswith(("cross_source_", "s2_s3_", "candidate_source_s"))
    }
    return {
        "all_features": list(FEATURE_NAMES),
        "name_only": [name for name in FEATURE_NAMES
                      if name.startswith("name_") and name not in cross_field_features],
        "without_address": [name for name in FEATURE_NAMES
                            if not any(name.startswith(prefix) for prefix in address_prefixes)
                            and name not in cross_field_features],
        "without_location": [name for name in FEATURE_NAMES
                             if not any(name.startswith(prefix) for prefix in location_prefixes)
                             and name not in {"city_name_consistency", "name_postal_consistency"}],
        "without_character": [name for name in FEATURE_NAMES if name not in character_features],
        "without_tfidf": [name for name in FEATURE_NAMES if name not in tfidf_features],
        "without_landmark": [name for name in FEATURE_NAMES if name != "address_landmark_overlap"],
        "without_cross_source": [name for name in FEATURE_NAMES
                                 if name not in cross_source_features],
    }


def evaluate_validation_model(model: Any,
                              val_df: pd.DataFrame,
                              truth: Mapping[str, Set[str]],
                              entity_ids: Sequence[str],
                              threshold: float,
                              score_margin: float | None,
                              candidate_recall_value: float,
                              feature_names: Sequence[str] = FEATURE_NAMES) -> Dict[str, float]:
    scored_pairs, probabilities = scored_pair_frame(model, val_df, feature_names)
    return evaluate_scored_pairs(
        scored_pairs, probabilities, truth, entity_ids, threshold, score_margin,
        candidate_recall_value,
    )


def evaluate_scored_pairs(scored_pairs: pd.DataFrame,
                          probabilities: np.ndarray,
                          truth: Mapping[str, Set[str]],
                          entity_ids: Sequence[str],
                          threshold: float,
                          score_margin: float | None,
                          candidate_recall_value: float) -> Dict[str, float]:
    predictions = predictions_from_pair_scores(scored_pairs, probabilities, threshold, score_margin, entity_ids)
    labels = scored_pairs["label"].astype(int).to_numpy() if not scored_pairs.empty else np.asarray([], dtype=int)
    selected = probabilities >= threshold
    if score_margin is not None and len(probabilities):
        maximum_by_entity: Dict[str, float] = {}
        for entity_id, probability in zip(scored_pairs["entity_id"].astype(str), probabilities):
            maximum_by_entity[entity_id] = max(maximum_by_entity.get(entity_id, -np.inf), float(probability))
        selected &= np.asarray([
            probability >= maximum_by_entity[str(entity_id)] - score_margin
            for entity_id, probability in zip(scored_pairs["entity_id"], probabilities)
        ])
    true_positive_pairs = int(np.count_nonzero(selected & (labels == 1)))
    false_positive_pairs = int(np.count_nonzero(selected & (labels == 0)))
    if len(labels):
        clipped_probabilities = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        pair_brier = float(np.mean((clipped_probabilities - labels) ** 2))
        pair_log_loss = float(-np.mean(
            labels * np.log(clipped_probabilities)
            + (1 - labels) * np.log(1.0 - clipped_probabilities)
        ))
    else:
        pair_brier = pair_log_loss = 0.0
    total_positive_pairs = sum(len(truth.get(str(entity_id), set())) for entity_id in entity_ids)
    pair_precision = (true_positive_pairs / (true_positive_pairs + false_positive_pairs)
                      if true_positive_pairs + false_positive_pairs else 1.0)
    pair_recall = true_positive_pairs / total_positive_pairs if total_positive_pairs else 1.0

    singleton_ids = [entity_id for entity_id in entity_ids if not truth.get(str(entity_id), set())]
    multi_match_ids = [entity_id for entity_id in entity_ids if len(truth.get(str(entity_id), set())) > 1]
    source2_truth = {str(entity_id): {candidate for candidate in truth.get(str(entity_id), set())
                                      if candidate.startswith("S2-")} for entity_id in entity_ids}
    source3_truth = {str(entity_id): {candidate for candidate in truth.get(str(entity_id), set())
                                      if candidate.startswith("S3-")} for entity_id in entity_ids}
    source2_predictions = {str(entity_id): {candidate for candidate in predictions.get(str(entity_id), set())
                                            if candidate.startswith("S2-")} for entity_id in entity_ids}
    source3_predictions = {str(entity_id): {candidate for candidate in predictions.get(str(entity_id), set())
                                            if candidate.startswith("S3-")} for entity_id in entity_ids}
    return {
        "candidate_pair_recall": candidate_recall_value,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_brier_score": pair_brier,
        "pair_log_loss": pair_log_loss,
        "entity_f0_5": score_entity_predictions(predictions, truth, entity_ids),
        "singleton_f0_5": score_entity_predictions(predictions, truth, singleton_ids),
        "multi_match_f0_5": score_entity_predictions(predictions, truth, multi_match_ids),
        "source2_f0_5": score_entity_predictions(source2_predictions, source2_truth, entity_ids),
        "source3_f0_5": score_entity_predictions(source3_predictions, source3_truth, entity_ids),
    }


def process_peak_memory_mb() -> float:
    try:
        import resource
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux returns KiB; macOS returns bytes.
        return peak / (1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0)
    except (ImportError, AttributeError, ValueError):
        return 0.0


def current_rss_mb() -> float:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return process_peak_memory_mb()


class PeakRssSampler:
    """Sample process resident memory during one ablation experiment."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.peak_mb = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.peak_mb = current_rss_mb()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.peak_mb = max(self.peak_mb, current_rss_mb())

    def stop(self) -> float:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self.peak_mb = max(self.peak_mb, current_rss_mb())
        return self.peak_mb


def run_ablation_suite(train_df: pd.DataFrame,
                       random_negative_train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       truth: Mapping[str, Set[str]],
                       entity_ids: Sequence[str],
                       model_types: Sequence[str],
                       seed: int,
                       candidate_recall_value: float,
                       output_path: Path,
                       train_source1: pd.DataFrame,
                       validation_source1: pd.DataFrame,
                       target_sources: pd.DataFrame,
                       source_lookup: pd.DataFrame,
                       negatives_per_positive: int,
                       random_negatives: int,
                       max_block_frequency: int) -> None:
    rows: List[Dict[str, object]] = []
    experiments: List[Tuple[str, pd.DataFrame, Sequence[str]]] = [
        (name, train_df, names) for name, names in feature_ablation_sets().items()
    ]
    experiments.append(("random_negatives_only", random_negative_train_df, FEATURE_NAMES))
    for model_type in model_types:
        for experiment_name, experiment_train_df, feature_names in experiments:
            started = time.perf_counter()
            memory_sampler = PeakRssSampler()
            memory_sampler.start()
            model, threshold, margin, _ = train_model(
                experiment_train_df, val_df, truth, entity_ids,
                seed=seed, model_type=model_type, feature_names=feature_names,
            )
            metrics = evaluate_validation_model(
                model, val_df, truth, entity_ids, threshold, margin, candidate_recall_value,
                feature_names,
            )
            gate_eval_baseline = ""
            gate_eval_after = ""
            if experiment_name == "all_features":
                _, gate_eval_baseline, gate_eval_after = train_entity_decision_layer(
                    model, val_df, truth, entity_ids, threshold, margin, seed
                )
            experiment_peak_rss_mb = memory_sampler.stop()
            rows.append({
                "experiment": experiment_name,
                "model": model_type,
                "feature_count": len(feature_names),
                "train_pairs": len(experiment_train_df),
                "validation_pairs": len(val_df),
                "threshold": threshold,
                "score_margin": "disabled" if margin is None else margin,
                "experiment_seconds": time.perf_counter() - started,
                "experiment_peak_rss_mb": experiment_peak_rss_mb,
                "gate_eval_entity_f0_5_before": gate_eval_baseline,
                "gate_eval_entity_f0_5_after": gate_eval_after,
                **metrics,
            })
            print(f"Ablation {experiment_name} ({model_type}): entity F0.5={metrics['entity_f0_5']:.4f}, "
                  f"pair P/R={metrics['pair_precision']:.4f}/{metrics['pair_recall']:.4f}")

        normalization_experiments = (
            ("without_legal_suffix_stripping", NormalizationSettings(False, True, True)),
            ("without_dba_alias_splitting", NormalizationSettings(True, False, True)),
            ("without_address_abbreviation_normalization", NormalizationSettings(True, True, False)),
        )
        for experiment_name, settings in normalization_experiments:
            started = time.perf_counter()
            memory_sampler = PeakRssSampler()
            memory_sampler.start()
            global NORMALIZATION_SETTINGS
            previous_settings = NORMALIZATION_SETTINGS
            try:
                NORMALIZATION_SETTINGS = settings
                variant_index = build_block_index(
                    target_sources, max_block_frequency=max_block_frequency
                )
                variant_train_df, _, _ = build_training_examples(
                    train_source1, None, truth,
                    negatives_per_positive=negatives_per_positive,
                    random_negatives=random_negatives,
                    training=True,
                    seed=seed,
                    source_lookup=source_lookup,
                    index=variant_index,
                )
                variant_val_df, variant_recall, _ = build_training_examples(
                    validation_source1, None, truth,
                    training=False,
                    seed=seed,
                    source_lookup=source_lookup,
                    index=variant_index,
                )
            finally:
                NORMALIZATION_SETTINGS = previous_settings

            model, threshold, margin, _ = train_model(
                variant_train_df, variant_val_df, truth, entity_ids,
                seed=seed, model_type=model_type,
            )
            metrics = evaluate_validation_model(
                model, variant_val_df, truth, entity_ids, threshold, margin, variant_recall,
            )
            experiment_peak_rss_mb = memory_sampler.stop()
            rows.append({
                "experiment": experiment_name,
                "model": model_type,
                "feature_count": len(FEATURE_NAMES),
                "train_pairs": len(variant_train_df),
                "validation_pairs": len(variant_val_df),
                "threshold": threshold,
                "score_margin": "disabled" if margin is None else margin,
                "experiment_seconds": time.perf_counter() - started,
                "experiment_peak_rss_mb": experiment_peak_rss_mb,
                "gate_eval_entity_f0_5_before": "",
                "gate_eval_entity_f0_5_after": "",
                **metrics,
            })
            print(f"Ablation {experiment_name} ({model_type}): entity F0.5={metrics['entity_f0_5']:.4f}, "
                  f"candidate recall={variant_recall:.4f}")
            del variant_train_df, variant_val_df, variant_index
            gc.collect()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_output_rows(path: Path, rows: List[Tuple[str, List[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matched_ids in rows:
            handle.write(f"{s1_id}\t{','.join(matched_ids)}\n")


def write_candidate_rows(path: Path, rows: List[Tuple[str, List[str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, candidate_ids in rows:
            handle.write(f"{s1_id}\t{','.join(candidate_ids)}\n")


def find_submission_validator(script_path: Path | None = None,
                              cwd: Path | None = None,
                              test_dir: Path | None = None) -> Path | None:
    """Find the challenge validator in a checkout or an unpacked workspace."""
    roots: List[Path] = []
    for start in (script_path or Path(__file__), cwd or Path.cwd(), test_dir):
        if start is None:
            continue
        start = start.resolve()
        if start.is_file():
            start = start.parent
        roots.extend((start, *start.parents))

    candidates: List[Path] = []
    for root in roots:
        candidates.extend((
            root / "student_resource" / "utils" / "validate_submission.py",
            root / "utils" / "validate_submission.py",
        ))
    seen: Set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    return None


def predict_test_set(test_source1: pd.DataFrame,
                     source2_source3_df: pd.DataFrame | None,
                     model: Any,
                     threshold: float,
                     score_margin: float | None = None,
                     max_block_frequency: int = 10000,
                     source_lookup: Mapping[str, object] | pd.DataFrame | None = None,
                     index: Dict[Tuple[str, str], Set[str]] | None = None,
                     entity_decision: EntityDecisionLayer | None = None,
                     probability_calibrator: Any | None = None,
                     ) -> Iterator[Tuple[str, List[str], List[str]]]:
    if source_lookup is None and source2_source3_df is not None:
        source_lookup = build_source_lookup(source2_source3_df)
    if index is None and source2_source3_df is not None:
        index = build_block_index(source2_source3_df, max_block_frequency=max_block_frequency)
    if source_lookup is None or index is None:
        raise ValueError("Provide the Source-2/3 lookup and block index, or the source dataframe.")

    for row in test_source1.itertuples(index=False):
        s1_values = row._asdict()
        s1_id = str(s1_values.get("entity_id", ""))
        candidates, block_candidates = generate_candidate_details(
            s1_values, index, source_lookup, collect_blocks=True
        )
        if not candidates:
            yield s1_id, candidates, []
            continue

        feature_rows = [build_pair_features(
            s1_values, lookup_record(source_lookup, candidate_id), index,
            candidate_rank=position + 1,
        ) for position, candidate_id in enumerate(candidates)]
        add_candidate_block_memberships(feature_rows, candidates, block_candidates,
                                        index if isinstance(index, BlockIndex) else None)
        first_stage_model = model.base_model if isinstance(model, NeuralReranker) else model
        base_probabilities = predict_feature_rows(first_stage_model, feature_rows)
        if isinstance(model, NeuralReranker):
            top_positions = sorted(
                range(len(candidates)),
                key=lambda position: (-base_probabilities[position], candidates[position]),
            )[:model.top_k]
            scored_positions = top_positions
            scored_feature_rows = [feature_rows[position] for position in scored_positions]
            raw_scoring_probabilities = predict_feature_rows(model.reranker, scored_feature_rows)
        else:
            scored_positions = list(range(len(candidates)))
            scored_feature_rows = feature_rows
            raw_scoring_probabilities = base_probabilities
        reported_candidates = [candidates[position] for position in scored_positions]
        scoring_probabilities = apply_probability_calibrator(
            probability_calibrator, raw_scoring_probabilities
        )

        predicted_cardinality = None
        if entity_decision is not None:
            entity_features = entity_decision_feature_vector(
                scored_feature_rows, raw_scoring_probabilities, pair_threshold=0.0
            )
            entity_features[ENTITY_FEATURE_NAMES.index("number_above_pair_threshold")] = float(
                np.count_nonzero(scoring_probabilities >= entity_decision.feature_pair_threshold)
            )
            if entity_decision.classifier is not None:
                linked_probability = apply_binary_score_calibrator(
                    entity_decision.probability_calibrator,
                    entity_decision.classifier.predict_proba([entity_features])[:, 1],
                )[0]
                if linked_probability < entity_decision.threshold:
                    yield s1_id, reported_candidates, []
                    continue
            if entity_decision.cardinality_classifier is not None:
                predicted_classes, _ = cardinality_labels_from_probabilities(
                    entity_decision.cardinality_classifier,
                    np.asarray([entity_features], dtype=float),
                    confidence_thresholds=entity_decision.cardinality_confidence_thresholds,
                    probability_calibrator=entity_decision.cardinality_probability_calibrator,
                )
                predicted_cardinality = int(predicted_classes[0])

        if len(scoring_probabilities):
            best_probability = float(scoring_probabilities.max())
            selected_positions = [
                position for position, probability in zip(scored_positions, scoring_probabilities)
                if probability >= threshold and (score_margin is None or
                                                  probability >= best_probability - score_margin)
            ]
        else:
            selected_positions = []
        if predicted_cardinality == 0:
            selected_positions = []
        elif predicted_cardinality == 1 and len(scoring_probabilities):
            best_position = scored_positions[int(np.argmax(scoring_probabilities))]
            cardinality_threshold = (entity_decision.cardinality_thresholds.get(1, threshold)
                                     if entity_decision is not None else threshold)
            selected_positions = ([best_position]
                                  if scoring_probabilities.max() >= cardinality_threshold else [])
        elif predicted_cardinality is not None and predicted_cardinality >= 2:
            cardinality_threshold = (entity_decision.cardinality_thresholds.get(2, threshold)
                                     if entity_decision is not None else threshold)
            cardinality_margin = (entity_decision.cardinality_margins.get(2, score_margin)
                                  if entity_decision is not None else score_margin)
            best_probability = (float(scoring_probabilities.max())
                                if len(scoring_probabilities) else -np.inf)
            selected_positions = [
                position for position, probability in zip(scored_positions, scoring_probabilities)
                if probability >= cardinality_threshold
                and (cardinality_margin is None
                     or probability >= best_probability - cardinality_margin)
            ]
        selected = sorted(candidates[position] for position in selected_positions)
        yield s1_id, reported_candidates, selected


def write_prediction_outputs(output_dir: Path,
                             root_output: Path,
                             predictions: Iterable[Tuple[str, List[str], List[str]]]) -> None:
    """Write both required TSVs incrementally so test predictions stay bounded in memory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    root_output.mkdir(parents=True, exist_ok=True)
    candidate_paths = list(dict.fromkeys([
        output_dir / "candidate_pairs.tsv", root_output / "candidate_pairs.tsv"
    ]))
    matching_paths = list(dict.fromkeys([
        output_dir / "matching_results.tsv", root_output / "matching_results.tsv"
    ]))
    handles = {path: path.open("w", encoding="utf-8")
               for path in dict.fromkeys(candidate_paths + matching_paths)}
    try:
        for path in candidate_paths:
            handles[path].write("source1_entity_id\tcandidate_entity_ids\n")
        for path in matching_paths:
            handles[path].write("source1_entity_id\tmatched_entity_ids\n")
        for entity_id, candidates, matches in predictions:
            candidate_line = f"{entity_id}\t{','.join(candidates)}\n"
            match_line = f"{entity_id}\t{','.join(matches)}\n"
            for path in candidate_paths:
                handles[path].write(candidate_line)
            for path in matching_paths:
                handles[path].write(match_line)
    finally:
        for handle in handles.values():
            handle.close()


def write_normalization_audit(path: Path,
                              source1: pd.DataFrame,
                              max_rows: int,
                              seed: int) -> None:
    audit_sample = source1
    if max_rows > 0 and len(source1) > max_rows:
        audit_sample = source1.sample(n=max_rows, random_state=seed)
    prepared: List[Dict[str, object]] = []
    raw_names_by_core: Dict[str, Set[str]] = defaultdict(set)
    for row in audit_sample.itertuples(index=False):
        values = row._asdict()
        raw_name = str(values.get("business_name", ""))
        raw_address = str(values.get("business_address", ""))
        country = values.get("country", "")
        core_name = " ".join(core_name_tokens(raw_name))
        if core_name:
            raw_names_by_core[core_name].add(raw_name)
        prepared.append({
            "entity_id": str(values.get("entity_id", "")),
            "country": str(country),
            "raw_business_name": raw_name,
            "normalized_business_name": normalize_text(raw_name),
            "core_name": core_name,
            "name_variants": " || ".join(" ".join(variant) for variant in name_variants(raw_name)),
            "raw_business_address": raw_address,
            "normalized_address_tokens": " ".join(address_tokens(raw_address)),
            "parsed_city_tokens": " ".join(address_city_tokens(raw_address, country)),
            "parsed_state_tokens": " ".join(address_state_tokens(raw_address, country)),
            "postal_codes": ",".join(sorted(postal_codes(raw_address))),
            "house_number": house_number(raw_address),
            "unit_numbers": ",".join(sorted(unit_numbers(raw_address))),
            "phone_fragments": ",".join(sorted(phone_fragments(raw_address))),
        })
    for row in prepared:
        core_name = str(row["core_name"])
        colliding_names = sorted(raw_names_by_core.get(core_name, set())) if core_name else []
        row["core_name_collision_count_in_sample"] = len(colliding_names)
        row["core_name_collision_examples"] = " || ".join(colliding_names[:5])

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(prepared[0]) if prepared else ["entity_id", "raw_business_name", "core_name"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(prepared)


def write_normalization_collision_report(path: Path,
                                         source_frames: Mapping[str, pd.DataFrame]) -> None:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {"records": 0, "raw_values": set()}
    )
    totals: Counter[str] = Counter()
    collided_records: Counter[str] = Counter()
    for source_name, frame in source_frames.items():
        for row in frame.itertuples(index=False):
            values = row._asdict()
            country_raw = str(values.get("country", ""))
            country = normalize_country(country_raw)
            raw_name = str(values.get("business_name", ""))
            raw_address = str(values.get("business_address", ""))
            normalized_values = {
                "normalized_name": (" ".join(token_list(raw_name)), raw_name),
                "core_name": (" ".join(core_name_tokens(raw_name)), raw_name),
                "normalized_address": (" ".join(address_tokens(raw_address)), raw_address),
            }
            for field_name, (key, raw_value) in normalized_values.items():
                if not key:
                    continue
                totals[field_name] += 1
                group = grouped[(field_name, country, key)]
                group["records"] += 1
                group["raw_values"].add(raw_value)
            if country:
                totals["country"] += 1
                group = grouped[("country", "", country)]
                group["records"] += 1
                group["raw_values"].add(country_raw)

    rows: List[Dict[str, object]] = []
    for field_name, total in totals.items():
        groups = [(key, details) for key, details in grouped.items()
                  if key[0] == field_name and len(details["raw_values"]) > 1]
        collided_records[field_name] = sum(int(details["records"]) for _, details in groups)
        rows.append({
            "record_type": "summary",
            "field": field_name,
            "country": "",
            "normalized_key": "",
            "record_count": total,
            "distinct_raw_value_count": "",
            "collision_rate": collided_records[field_name] / total if total else 0.0,
            "raw_examples": f"{len(groups)} normalized keys map to multiple raw forms",
        })
    for (field_name, country, normalized_key), details in grouped.items():
        raw_values = sorted(details["raw_values"])
        if len(raw_values) < 2:
            continue
        rows.append({
            "record_type": "collision",
            "field": field_name,
            "country": country,
            "normalized_key": normalized_key,
            "record_count": details["records"],
            "distinct_raw_value_count": len(raw_values),
            "collision_rate": "",
            "raw_examples": " || ".join(raw_values[:8]),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        columns = ["record_type", "field", "country", "normalized_key", "record_count",
                   "distinct_raw_value_count", "collision_rate", "raw_examples"]
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_country_open_set_stress_report(path: Path,
                                         source1: pd.DataFrame,
                                         target_sources: pd.DataFrame,
                                         source_lookup: pd.DataFrame,
                                         index: BlockIndex,
                                         truth: Mapping[str, Set[str]],
                                         rows_per_scenario: int,
                                         seed: int,
                                         max_block_frequency: int,
                                         name_ngram_limit: int,
                                         channel_limit_multiplier: float,
                                         retrieval_context_mode: str,
                                         semantic_max_documents: int = 100000) -> pd.DataFrame:
    """Measure retrieval degradation when country labels are hidden or corrupted."""
    country_values = source1["country"].map(normalize_country)
    scenarios: List[Tuple[str, pd.DataFrame, Set[str] | None, str]] = []
    france = source1.loc[country_values == "france"].copy()
    us_india = source1.loc[country_values.isin({"us", "india"})].copy()
    if not france.empty:
        scenarios.append(("france_country_hidden", france, {"france"}, "mask_both"))
    if not us_india.empty:
        scenarios.append(("us_india_country_hidden", us_india, {"us", "india"}, "mask_both"))
        scenarios.append(("us_india_query_country_corrupted", us_india, {"us", "india"}, "corrupt_query"))
    scenarios.append(("all_country_hidden", source1.copy(), None, "mask_both"))

    output_rows: List[Dict[str, object]] = []
    for scenario_position, (scenario, query_rows, selected_countries, corruption) in enumerate(scenarios):
        if rows_per_scenario > 0 and len(query_rows) > rows_per_scenario:
            query_rows = query_rows.sample(n=rows_per_scenario, random_state=seed + scenario_position)
        query_rows = query_rows.copy().reset_index(drop=True)
        entity_ids = query_rows["entity_id"].astype(str).tolist()
        scenario_truth = {entity_id: truth.get(entity_id, set()) for entity_id in entity_ids}
        _, baseline_recall, baseline_diagnostics = build_training_examples(
            query_rows, None, scenario_truth, training=False, seed=seed,
            source_lookup=source_lookup, index=index,
        )
        stressed_queries = query_rows.copy()
        stressed_targets = target_sources
        masked_target_count = 0
        stressed_index = index
        if corruption == "mask_both":
            stressed_queries["country"] = ""
            stressed_targets = target_sources.copy()
            target_mask = (pd.Series(True, index=target_sources.index) if selected_countries is None
                           else target_sources["country"].map(normalize_country).isin(selected_countries))
            masked_target_count = int(target_mask.sum())
            stressed_targets.loc[target_mask, "country"] = ""
            stressed_index = build_block_index(
                stressed_targets, max_block_frequency=max_block_frequency,
                name_ngram_limit=name_ngram_limit,
                channel_limit_multiplier=channel_limit_multiplier,
                retrieval_context_mode=retrieval_context_mode,
                semantic_retrieval=index.semantic_vectorizer is not None,
                semantic_max_documents=semantic_max_documents,
                semantic_top_k=index.semantic_top_k,
                semantic_min_similarity=index.semantic_min_similarity,
            )
            stressed_lookup = build_source_lookup(stressed_targets)
        else:
            stressed_queries["country"] = stressed_queries["country"].map(
                lambda value: f"{normalize_country(value)}x" if normalize_country(value) else "unknown"
            )
            stressed_lookup = source_lookup
        _, stressed_recall, stressed_diagnostics = build_training_examples(
            stressed_queries, None, scenario_truth, training=False, seed=seed,
            source_lookup=stressed_lookup, index=stressed_index,
        )
        total_truth_pairs = stressed_diagnostics.total_truth_pairs
        record: Dict[str, object] = {
            "scenario": scenario,
            "source1_entities": len(query_rows),
            "target_records_country_masked": masked_target_count,
            "truth_pairs": total_truth_pairs,
            "baseline_candidate_recall": baseline_recall,
            "stressed_candidate_recall": stressed_recall,
            "candidate_recall_delta": stressed_recall - baseline_recall,
            "baseline_mean_candidates_at_200": (
                baseline_diagnostics.candidates_at_k[200] / max(baseline_diagnostics.entity_count, 1)
            ),
            "stressed_mean_candidates_at_200": (
                stressed_diagnostics.candidates_at_k[200] / max(stressed_diagnostics.entity_count, 1)
            ),
        }
        for limit in CANDIDATE_EFFICIENCY_K:
            baseline_at_k = (baseline_diagnostics.recall_at_k_hits[limit] / total_truth_pairs
                             if total_truth_pairs else 1.0)
            stressed_at_k = (stressed_diagnostics.recall_at_k_hits[limit] / total_truth_pairs
                             if total_truth_pairs else 1.0)
            record[f"baseline_recall_at_{limit}"] = baseline_at_k
            record[f"stressed_recall_at_{limit}"] = stressed_at_k
            record[f"recall_at_{limit}_delta"] = stressed_at_k - baseline_at_k
        output_rows.append(record)
        if stressed_index is not index:
            del stressed_index
            del stressed_lookup
            del stressed_targets
            gc.collect()

    report = pd.DataFrame(output_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(f"Country stress test written to {path}")
    for row in output_rows:
        print(f"  {row['scenario']}: candidate recall "
              f"{row['baseline_candidate_recall']:.4f} -> {row['stressed_candidate_recall']:.4f} "
              f"(delta {row['candidate_recall_delta']:+.4f})")
    return report


def diagnose_error_buckets(features: Mapping[str, float],
                           expected: bool,
                           predicted: bool) -> List[str]:
    value = lambda name: float(features.get(name, 0.0))
    buckets: List[str] = []
    name_exact = value("name_core_exact") >= 0.5
    address_similarity = value("address_token_jaccard")
    name_similarity = value("name_token_jaccard")
    postal_conflict = value("postal_code_conflict") >= 0.5
    house_conflict = value("house_number_conflict") >= 0.5
    country_conflict = value("country_conflict") >= 0.5

    if name_exact and (address_similarity < 0.5 or postal_conflict or house_conflict):
        buckets.append("same_name_different_address")
    if ((value("address_exact") >= 0.5 or address_similarity >= 0.8)
            and name_similarity < 0.5):
        buckets.append("same_address_different_name")
    if name_exact and (postal_conflict or house_conflict):
        buckets.append("franchise")
    if name_exact and value("common_name_tokens") <= 1:
        buckets.append("generic_name")
    if postal_conflict:
        buckets.append("postal_conflict")
    if house_conflict:
        buckets.append("house_conflict")
    if country_conflict:
        buckets.append("country_conflict")
    if name_exact and value("name_exact") < 0.5:
        buckets.append("normalization_collision")
    if predicted and not expected and (
            value("cross_source_candidate_support_count") > 0
            or value("cross_source_direct_edge_count") > 0):
        buckets.append("graph_false_support")

    if expected:
        if value("name_variant_similarity") >= 0.8 and not name_exact:
            buckets.append("dba")
        if 0.55 <= value("name_char_similarity") < 0.95 and name_similarity > 0:
            buckets.append("typo")
        if (value("name_tfidf_cosine") >= 0.35
                and value("name_char_similarity") < 0.5 and name_similarity > 0):
            buckets.append("transliteration")
        if value("has_address_both") < 0.5:
            buckets.append("missing_address")
        if value("has_name_both") < 0.5:
            buckets.append("missing_name")
        if (address_similarity >= 0.5
                and value("address_char_similarity") + 0.15 < address_similarity):
            buckets.append("address_reorder")
        if (value("city_token_jaccard") == 0
                and (address_similarity >= 0.5 or value("postal_code_match") > 0)):
            buckets.append("city_extraction_failure")
        if (value("same_country") > 0 and value("has_address_both") > 0
                and value("address_layout_country_plausibility") <= 0.1):
            buckets.append("unseen_country_pattern")
    return list(dict.fromkeys(buckets or ["other_candidate"]))


def write_oof_error_report(path: Path,
                           bucket_path: Path,
                           result: OOFModelResult,
                           truth: Mapping[str, Set[str]],
                           threshold: float,
                           score_margin: float | None,
                           entity_decision: EntityDecisionLayer | None) -> None:
    entity_ids = [entity_id for position, entity_id in enumerate(result.entity_ids)
                  if result.fold_by_entity[position] == 0]
    oof_rows = result.pair_rows[result.pair_rows["fold"].astype(int) == 0].reset_index(drop=True)
    probabilities = apply_probability_calibrator(
        result.probability_calibrator,
        oof_rows["probability"].to_numpy(dtype=float),
    )
    predictions = predictions_from_pair_scores(
        oof_rows, probabilities, threshold, score_margin, entity_ids
    )
    gate_features = result.entity_features[
        [position for position, fold in enumerate(result.fold_by_entity) if fold == 0]
    ]
    if entity_decision is not None and entity_decision.classifier is not None:
        gate_probabilities = apply_binary_score_calibrator(
            entity_decision.probability_calibrator,
            entity_decision.classifier.predict_proba(gate_features)[:, 1],
        )
        predictions = {
            entity_id: predictions.get(entity_id, set())
            if probability >= entity_decision.threshold else set()
            for entity_id, probability in zip(entity_ids, gate_probabilities)
        }
    if entity_decision is not None and entity_decision.cardinality_classifier is not None:
        cardinality, _ = cardinality_labels_from_probabilities(
            entity_decision.cardinality_classifier,
            gate_features,
            confidence_thresholds=entity_decision.cardinality_confidence_thresholds,
            probability_calibrator=entity_decision.cardinality_probability_calibrator,
        )
        predictions = cardinality_adjusted_predictions(
            oof_rows, probabilities, threshold, score_margin, entity_ids, cardinality,
            entity_decision.cardinality_thresholds,
            entity_decision.cardinality_margins,
        )
        if entity_decision.classifier is not None:
            predictions = {
                entity_id: predictions.get(entity_id, set())
                if probability >= entity_decision.threshold else set()
                for entity_id, probability in zip(entity_ids, gate_probabilities)
            }

    score_by_pair = {
        (str(row.entity_id), str(row.candidate_id)): float(probability)
        for row, probability in zip(oof_rows.itertuples(index=False), probabilities)
    }
    base_score_by_pair = {
        (str(row.entity_id), str(row.candidate_id)): float(probability)
        for row, probability in zip(
            result.holdout_examples.itertuples(index=False), result.holdout_base_probabilities
        )
    }
    candidate_set_by_entity: Dict[str, Set[str]] = defaultdict(set)
    prepared: List[Dict[str, object]] = []
    bucket_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"count": 0, "true_positive": 0, "false_positive": 0,
                 "false_negative": 0}
    )
    bucket_entities: Dict[str, Set[str]] = defaultdict(set)
    bucket_truth: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    bucket_predictions: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    bucket_errors: Dict[str, List[Tuple[str, str, bool, bool]]] = defaultdict(list)
    display_features = (
        "name_exact", "name_core_exact", "common_name_tokens", "name_char_similarity",
        "name_tfidf_cosine", "name_variant_similarity", "name_token_jaccard",
        "address_exact", "address_token_jaccard", "address_char_similarity",
        "has_name_both", "has_address_both", "address_layout_country_plausibility",
        "postal_code_match", "house_number_match", "city_token_jaccard", "state_match",
        "same_country", "country_conflict", "postal_code_conflict", "house_number_conflict",
        "contradiction_score", "phone_fragment_match", "candidate_retrieval_rank_reciprocal",
        "candidate_retrieval_score", "candidate_block_name_exact", "candidate_block_name_pair",
        "candidate_block_name_phonetic",
        "candidate_block_name_tfidf",
        "candidate_block_name_char_ngram", "candidate_block_name_token",
        "candidate_block_address_token", "candidate_block_postal",
        "candidate_block_house_number", "candidate_block_city_token",
        "candidate_retrieval_channel_count", "cross_source_candidate_support_count",
        "cross_source_direct_edge_count",
    )
    for row in result.holdout_examples.itertuples(index=False):
        entity_id, candidate_id = str(row.entity_id), str(row.candidate_id)
        candidate_set_by_entity[entity_id].add(candidate_id)
        expected = candidate_id in truth.get(entity_id, set())
        predicted = candidate_id in predictions.get(entity_id, set())
        row_buckets = diagnose_error_buckets(row.features, expected, predicted)
        for bucket in row_buckets:
            bucket_entities[bucket].add(entity_id)
            bucket_stats[bucket]["count"] += 1
            bucket_stats[bucket]["true_positive"] += int(expected and predicted)
            bucket_stats[bucket]["false_positive"] += int(not expected and predicted)
            bucket_stats[bucket]["false_negative"] += int(expected and not predicted)
            if expected:
                bucket_truth[bucket][entity_id].add(candidate_id)
            if predicted:
                bucket_predictions[bucket][entity_id].add(candidate_id)
            if expected != predicted:
                bucket_errors[bucket].append((entity_id, candidate_id, expected, predicted))
        if expected == predicted:
            continue
        features = row.features
        scored_probability = score_by_pair.get((entity_id, candidate_id))
        base_probability = base_score_by_pair.get((entity_id, candidate_id))
        score_stage = ("calibrated_pair_model" if scored_probability is not None else
                       "first_stage_only" if base_probability is not None else "not_scored")
        error_type = "false_negative" if expected else "false_positive"
        if expected and scored_probability is None and result.model_type.endswith("_mlp"):
            error_type = "false_negative_topk_excluded"
        prepared.append({
            "error_type": error_type,
            "source1_entity_id": entity_id,
            "candidate_entity_id": candidate_id,
            "pair_probability": (scored_probability if scored_probability is not None else
                                 base_probability if base_probability is not None else ""),
            "score_stage": score_stage,
            "true_match": int(expected),
            "predicted_match": int(predicted),
            "error_buckets": "|".join(row_buckets),
            **{name: features.get(name, "") for name in display_features},
        })
    for entity_id in entity_ids:
        for candidate_id in truth.get(entity_id, set()) - candidate_set_by_entity.get(entity_id, set()):
            bucket_entities["blocker_miss"].add(entity_id)
            bucket_stats["blocker_miss"]["count"] += 1
            bucket_stats["blocker_miss"]["false_negative"] += 1
            bucket_truth["blocker_miss"][entity_id].add(candidate_id)
            bucket_errors["blocker_miss"].append((entity_id, candidate_id, True, False))
            prepared.append({
                "error_type": "candidate_retrieval_false_negative",
                "source1_entity_id": entity_id,
                "candidate_entity_id": candidate_id,
                "pair_probability": "",
                "score_stage": "not_retrieved",
                "true_match": 1,
                "predicted_match": 0,
                "error_buckets": "blocker_miss",
                **{name: "" for name in display_features},
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["error_type", "source1_entity_id", "candidate_entity_id", "pair_probability",
               "score_stage", "true_match", "predicted_match", "error_buckets",
               *display_features]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(prepared)

    baseline_score = score_entity_predictions(predictions, truth, entity_ids)
    summary_rows: List[Dict[str, object]] = []
    # Keep the catch-all visible even when this holdout happens to contain none.
    _ = bucket_stats["other_candidate"]
    for bucket, counts in sorted(bucket_stats.items()):
        true_positives = counts["true_positive"]
        false_positives = counts["false_positive"]
        false_negatives = counts["false_negative"]
        predicted_count = true_positives + false_positives
        actual_count = true_positives + false_negatives
        bucket_score = score_entity_predictions(
            bucket_predictions[bucket], bucket_truth[bucket], bucket_entities[bucket]
        )
        corrected = {entity_id: set(values) for entity_id, values in predictions.items()}
        for entity_id, candidate_id, expected, predicted in bucket_errors[bucket]:
            if predicted and not expected:
                corrected.setdefault(entity_id, set()).discard(candidate_id)
            elif expected and not predicted:
                corrected.setdefault(entity_id, set()).add(candidate_id)
        corrected_score = score_entity_predictions(corrected, truth, entity_ids)
        summary_rows.append({
            "bucket": bucket,
            "count": counts["count"],
            "share_of_bucketed_candidates": 0.0,
            "true_positive": true_positives,
            "false_positive": false_positives,
            "false_negative": false_negatives,
            "precision": true_positives / predicted_count if predicted_count else 1.0,
            "recall": true_positives / actual_count if actual_count else 1.0,
            "bucket_entity_f0_5": bucket_score,
            "overall_f0_5_if_bucket_errors_fixed": corrected_score,
            "f0_5_impact_if_fixed": corrected_score - baseline_score,
        })
    total_bucket_assignments = sum(int(row["count"]) for row in summary_rows)
    for row in summary_rows:
        row["share_of_bucketed_candidates"] = (
            int(row["count"]) / total_bucket_assignments if total_bucket_assignments else 0.0
        )
    summary_rows.sort(key=lambda row: (
        row["bucket"] != "other_candidate", -int(row["count"]), str(row["bucket"])
    ))
    other_count = int(bucket_stats.get("other_candidate", {}).get("count", 0))
    other_share = other_count / total_bucket_assignments if total_bucket_assignments else 0.0
    print(f"OOF error catch-all other_candidate: {other_count} rows "
          f"({other_share:.2%} of bucket assignments).")
    bucket_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows, columns=[
        "bucket", "count", "share_of_bucketed_candidates",
        "true_positive", "false_positive", "false_negative",
        "precision", "recall", "bucket_entity_f0_5",
        "overall_f0_5_if_bucket_errors_fixed", "f0_5_impact_if_fixed",
    ]).to_csv(bucket_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="student_resource/output")
    parser.add_argument("--sample-train-rows", type=int, default=0,
                        help="Stratified Source-1 sample for OOF model selection; 0 uses all training entities.")
    parser.add_argument("--final-train-rows", type=int, default=0,
                        help="Stratified Source-1 sample for the final pair model; 0 uses all training entities.")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of entity-level OOF folds; fold index 0 is reserved for independent evaluation.")
    parser.add_argument("--negatives-per-positive", type=int, default=20)
    parser.add_argument("--random-negatives", type=int, default=5)
    parser.add_argument("--adversarial-negatives-per-entity", type=int, default=5,
                        help="Mine up to this many hard negatives per entity in OOF and each of two final rounds; 0 disables.")
    parser.add_argument("--model", choices=("logistic", "lightgbm", "compare", "ensemble"), default="lightgbm",
                        help="Pair matcher; compare evaluates Logistic Regression and LightGBM; ensemble also evaluates their probability average.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Add an OOF-validated probability-average ensemble of logistic and LightGBM models.")
    parser.add_argument("--neural-reranker", action="store_true",
                        help="Fit an opt-in MLP reranker over the first-stage model's top-K pairs per entity.")
    parser.add_argument("--neural-top-k", type=int, default=20,
                        help="Maximum candidate pairs per entity passed to the optional MLP reranker.")
    parser.add_argument("--run-ablations", action="store_true",
                        help="Fit feature, negative-sampling, normalization, and gate ablations; write ablation_results.csv.")
    parser.add_argument("--normalization-audit", action="store_true",
                        help="Write a sampled normalization review to normalization_audit.tsv.")
    parser.add_argument("--normalization-audit-rows", type=int, default=5000,
                        help="Maximum rows in the normalization audit; use 0 for all Source-1 rows.")
    parser.add_argument("--max-block-frequency", type=int, default=10000,
                        help="Base block frequency ceiling, scaled by block type; use 0 to retain all postings.")
    parser.add_argument("--name-ngram-limit", type=int, default=6,
                        help="Maximum sampled name character n-grams; use 0 to keep all n-grams.")
    parser.add_argument("--candidate-channel-limit-multiplier", type=float, default=1.0,
                        help="Scale each retrieval channel's candidate cap by this factor.")
    parser.add_argument("--retrieval-context-mode", choices=("once", "per_channel"), default="once",
                        help="Add retrieval context once per candidate or once per channel.")
    parser.add_argument("--disable-target-graph", action="store_true",
                        help="Skip the memory-intensive cross-source target graph; pair graph features are zero.")
    parser.add_argument("--semantic-retrieval", action="store_true",
                        help="Enable a gated character-TF-IDF candidate fallback for sparse symbolic retrieval.")
    parser.add_argument("--semantic-index-max-targets", type=int, default=100000,
                        help="Safety limit for target rows in the optional character-TF-IDF index.")
    parser.add_argument("--semantic-top-k", type=int, default=50,
                        help="Maximum fuzzy-name candidates added when symbolic retrieval is sparse.")
    parser.add_argument("--semantic-min-similarity", type=float, default=0.12,
                        help="Minimum character-TF-IDF cosine score for fallback candidates.")
    parser.add_argument("--run-retrieval-ablation", action="store_true",
                        help="Measure n-gram caps, channel multipliers, and both context modes; write retrieval_ablation.csv.")
    parser.add_argument("--run-country-stress-test", action="store_true",
                        help="Measure candidate recall after masking France, US/India, or all country labels.")
    parser.add_argument("--country-stress-rows", type=int, default=1000,
                        help="Maximum Source-1 rows per country stress scenario; 0 uses every eligible row.")
    parser.add_argument("--compare-class-weight", action="store_true",
                        help="Add unweighted pair models to the same OOF comparison.")
    parser.add_argument("--compare-source-specific", action="store_true",
                        help="Add separate S1→S2 and S1→S3 pair models to the OOF comparison.")
    parser.add_argument("--run-experiment-matrix", action="store_true",
                        help="Run a 10-variant weighted/unweighted, shared/source-specific, and MLP OOF matrix; write oof_model_comparison.csv.")
    parser.add_argument("--validate-submission", action="store_true",
                        help="Run the challenge validator after writing outputs when it is available.")
    parser.add_argument("--skip-submission-validation", action="store_true",
                        help="Deprecated compatibility option; validation is skipped by default.")
    parser.add_argument("--check-submission-ids", action="store_true",
                        help="Ask the validator to load test S2/S3 IDs and check ID existence (memory intensive).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.skip_submission_validation and (args.validate_submission or args.check_submission_ids):
        parser.error("--skip-submission-validation cannot be combined with validator options.")

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cv_folds < 3:
        parser.error("--cv-folds must be at least 3.")
    if args.sample_train_rows < 0 or args.final_train_rows < 0:
        parser.error("--sample-train-rows and --final-train-rows cannot be negative.")
    if args.neural_top_k < 1:
        parser.error("--neural-top-k must be at least 1.")
    if args.country_stress_rows < 0:
        parser.error("--country-stress-rows cannot be negative.")
    if args.adversarial_negatives_per_entity < 0:
        parser.error("--adversarial-negatives-per-entity cannot be negative.")
    if args.name_ngram_limit < 0:
        parser.error("--name-ngram-limit cannot be negative.")
    if args.candidate_channel_limit_multiplier <= 0:
        parser.error("--candidate-channel-limit-multiplier must be positive.")
    if args.semantic_index_max_targets < 1 or args.semantic_top_k < 1:
        parser.error("Semantic index document and top-K limits must be positive.")
    if not 0.0 <= args.semantic_min_similarity <= 1.0:
        parser.error("--semantic-min-similarity must be between 0 and 1.")
    if args.semantic_index_max_targets < 1 or args.semantic_top_k < 1:
        parser.error("Semantic index document and top-K limits must be positive.")
    if not 0.0 <= args.semantic_min_similarity <= 1.0:
        parser.error("--semantic-min-similarity must be between 0 and 1.")

    train_s1 = load_source(str(train_dir / "train_source1.tsv"))
    train_s2 = load_source(str(train_dir / "train_source2.tsv"))
    train_s3 = load_source(str(train_dir / "train_source3.tsv"))
    truth = parse_truth(str(train_dir / "train_ground_truth.tsv"))
    print(f"Train Source1 rows: {len(train_s1)}")
    print(f"Train Source2 rows: {len(train_s2)}")
    print(f"Train Source3 rows: {len(train_s3)}")
    print(f"Positive Source1 entities: {sum(bool(value) for value in truth.values())}")
    print(f"Singletons in training: {sum(not value for value in truth.values())}")
    phone_address_count = sum(bool(phone_fragments(value)) for value in train_s1["business_address"])
    phone_address_rate = phone_address_count / len(train_s1) if len(train_s1) else 0.0
    print(f"Phone fragments in Source-1 addresses: {phone_address_count}/{len(train_s1)} "
          f"({phone_address_rate:.2%})")
    if args.normalization_audit:
        audit_path = output_dir / "normalization_audit.tsv"
        write_normalization_audit(audit_path, train_s1, args.normalization_audit_rows, args.seed)
        collision_path = output_dir / "normalization_collisions.tsv"
        write_normalization_collision_report(
            collision_path,
            {"S1": train_s1, "S2": train_s2, "S3": train_s3},
        )
        print(f"Wrote normalization sample to {audit_path}")
        print(f"Wrote normalization collision summary to {collision_path}")

    full_name_frequencies: Counter[str] = Counter()
    for value in train_s1["business_name"]:
        name = " ".join(core_name_tokens(value))
        if name:
            full_name_frequencies[name] += 1
    oof_source1 = stratified_sample_source1(
        train_s1, truth, args.sample_train_rows, full_name_frequencies, args.seed
    )
    print(f"OOF Source-1 entities: {len(oof_source1)} of {len(train_s1)}")

    print("Building training-source lookup and adaptive block index...")
    combined_train_sources = pd.concat([train_s2, train_s3], ignore_index=True, sort=False)
    train_lookup = build_source_lookup(combined_train_sources)
    if args.run_retrieval_ablation:
        retrieval_report = output_dir / "retrieval_ablation.csv"
        run_retrieval_ablation(
            oof_source1, truth, combined_train_sources, train_lookup, args.max_block_frequency,
            args.name_ngram_limit, args.candidate_channel_limit_multiplier,
            args.retrieval_context_mode,
            retrieval_report,
        )
        print(f"Wrote retrieval profile comparison to {retrieval_report}")
    train_index = build_block_index(
        combined_train_sources,
        max_block_frequency=args.max_block_frequency,
        name_ngram_limit=args.name_ngram_limit,
        channel_limit_multiplier=args.candidate_channel_limit_multiplier,
        retrieval_context_mode=args.retrieval_context_mode,
        build_graph=not args.disable_target_graph,
        semantic_retrieval=args.semantic_retrieval,
        semantic_max_documents=args.semantic_index_max_targets,
        semantic_top_k=args.semantic_top_k,
        semantic_min_similarity=args.semantic_min_similarity,
    )
    # The lookup and index own the target data needed downstream. Drop the original
    # source frames and concatenated frame before building pair features so large
    # runs do not keep several redundant copies of millions of target rows alive.
    del train_s2, train_s3
    if args.run_country_stress_test:
        write_country_open_set_stress_report(
            output_dir / "country_open_set_stress.csv", oof_source1,
            combined_train_sources, train_lookup, train_index, truth,
            args.country_stress_rows, args.seed, args.max_block_frequency,
            args.name_ngram_limit, args.candidate_channel_limit_multiplier,
            args.retrieval_context_mode, args.semantic_index_max_targets,
        )
    del combined_train_sources
    gc.collect()
    if args.run_experiment_matrix:
        if (args.ensemble or args.model == "ensemble") and LGBMClassifier is None:
            parser.error("--ensemble requires LightGBM; install the pinned requirements.txt.")
        model_families = ["logistic", "lightgbm"]
        model_types = [
            "logistic", "lightgbm",
            "logistic_unweighted", "lightgbm_unweighted",
            "logistic_source_specific", "lightgbm_source_specific",
            "logistic_unweighted_source_specific", "lightgbm_unweighted_source_specific",
            "logistic_mlp", "lightgbm_mlp",
        ]
        if args.ensemble or args.model == "ensemble":
            model_types.append("ensemble:logistic,lightgbm")
    else:
        run_ensemble = bool(args.ensemble or args.model == "ensemble")
        if run_ensemble and LGBMClassifier is None:
            parser.error("--ensemble requires LightGBM; install the pinned requirements.txt.")
        model_families = (["logistic", "lightgbm"]
                          if args.model in {"compare", "ensemble"} or run_ensemble else [args.model])
        model_types = list(model_families)
        if run_ensemble:
            model_types.append("ensemble:logistic,lightgbm")
        if args.compare_class_weight:
            model_types.extend(f"{model_type}_unweighted" for model_type in model_families)
        if args.compare_source_specific:
            source_variants = list(model_families)
            if args.compare_class_weight:
                source_variants.extend(f"{model_type}_unweighted" for model_type in model_families)
            model_types.extend(f"{model_type}_source_specific" for model_type in source_variants)
        if args.neural_reranker:
            model_types.extend(f"{model_type}_mlp" for model_type in model_families)
    print(f"Running {args.cv_folds}-fold entity OOF validation for: {', '.join(model_types)}")
    oof_results, sampled_training_examples, train_candidate_recall, train_candidate_diagnostics = (
        run_entity_oof_validation(
            oof_source1, truth, train_lookup, train_index, model_types,
            args.cv_folds, args.seed, args.negatives_per_positive, args.random_negatives,
            args.neural_top_k, args.adversarial_negatives_per_entity,
        )
    )
    selected_oof = calibrate_and_compare_oof_models(oof_results, truth, args.cv_folds)
    model_report_path = output_dir / "oof_model_comparison.csv"
    write_oof_model_comparison_report(model_report_path, oof_results, selected_oof.model_type)
    print(f"Wrote OOF model ranking to {model_report_path}")
    threshold = selected_oof.threshold
    score_margin = selected_oof.score_margin
    entity_decision, pair_only_holdout_score, gate_holdout_score, cardinality_holdout_score = (
        train_oof_entity_decision_layers(
            selected_oof, truth, args.cv_folds, threshold, score_margin
        )
    )
    print(f"Selected matcher: {selected_oof.model_type}; OOF CV entity F0.5="
          f"{selected_oof.cross_validation_f0_5:.4f}; reserved independent fold entity F0.5="
          f"{selected_oof.heldout_metrics.get('entity_f0_5', 0.0):.4f}")
    print(f"Reserved independent fold decision scores: pair={pair_only_holdout_score:.4f}, "
          f"gate={gate_holdout_score:.4f}, cardinality={cardinality_holdout_score:.4f}")
    if entity_decision is None:
        print("No entity gate or cardinality layer improved the tuning fold.")
    else:
        print(f"OOF decision layer active: linked gate={entity_decision.classifier is not None}; "
              f"cardinality model={entity_decision.cardinality_classifier is not None}")

    if args.run_ablations:
        calibration_mask = selected_oof.fold_by_entity != 0
        calibration_source1 = oof_source1.iloc[np.flatnonzero(calibration_mask)].copy()
        calibration_ids = set(calibration_source1["entity_id"].astype(str))
        calibration_training_examples = sampled_training_examples[
            sampled_training_examples["entity_id"].astype(str).isin(calibration_ids)
        ]
        holdout_source1 = selected_oof.holdout_source1
        holdout_examples = selected_oof.holdout_examples
        random_negative_examples, _, _ = build_training_examples(
            calibration_source1, None, truth,
            negatives_per_positive=args.negatives_per_positive,
            random_negatives=args.random_negatives,
            training=True,
            mine_hard_negatives=False,
            seed=args.seed,
            source_lookup=train_lookup,
            index=train_index,
        )
        report_path = output_dir / "ablation_results.csv"
        run_ablation_suite(
            calibration_training_examples, random_negative_examples, holdout_examples,
            truth, holdout_source1["entity_id"].astype(str).tolist(), model_families,
            args.seed, candidate_recall_for_rows(
                holdout_examples, truth, holdout_source1["entity_id"].astype(str).tolist()
            ), report_path,
            calibration_source1, holdout_source1, train_lookup.reset_index(), train_lookup,
            args.negatives_per_positive, args.random_negatives, args.max_block_frequency,
        )
        print(f"Wrote OOF calibration ablations to {report_path}")
        del random_negative_examples, calibration_training_examples, calibration_source1

    error_report_path = output_dir / "oof_validation_errors.tsv"
    error_bucket_path = output_dir / "oof_error_buckets.csv"
    write_oof_error_report(
        error_report_path, error_bucket_path, selected_oof, truth,
        threshold, score_margin, entity_decision
    )
    print(f"Wrote independent OOF error examples to {error_report_path} and bucket summary to {error_bucket_path}")

    final_source1 = stratified_sample_source1(
        train_s1, truth, args.final_train_rows, full_name_frequencies, args.seed + 1
    )
    final_source1_ids = set(final_source1["entity_id"].astype(str))
    oof_source1_ids = set(oof_source1["entity_id"].astype(str))
    final_training_examples = sampled_training_examples
    if final_source1_ids != oof_source1_ids:
        final_training_examples, _, final_diagnostics = build_training_examples(
            final_source1, None, truth,
            negatives_per_positive=args.negatives_per_positive,
            random_negatives=args.random_negatives,
            training=True,
            seed=args.seed,
            source_lookup=train_lookup,
            index=train_index,
        )
        del final_diagnostics
    mined_negatives = selected_oof.mined_negatives
    if not mined_negatives.empty:
        mined_negatives = mined_negatives[
            mined_negatives["entity_id"].astype(str).isin(final_source1_ids)
        ]
    if not mined_negatives.empty:
        final_training_examples = pd.concat(
            [final_training_examples, mined_negatives[["entity_id", "candidate_id", "label", "features"]]],
            ignore_index=True,
        ).drop_duplicates(["entity_id", "candidate_id"], keep="first")
    print(f"Final pair-model training pairs ({len(final_source1):,} labeled S1 entities): "
          f"{len(final_training_examples):,}")
    model = fit_final_pair_model(
        final_training_examples, selected_oof.model_type, args.seed, args.neural_top_k
    )
    for mining_round in range(2):
        round_negatives = mine_final_adversarial_negatives(
            final_source1, truth, train_lookup, train_index, model,
            final_training_examples, args.adversarial_negatives_per_entity,
            seed=args.seed + mining_round,
        )
        if round_negatives.empty:
            print(f"Final adversarial mining round {mining_round + 1} found no unseen negatives.")
            break
        final_training_examples = pd.concat(
            [final_training_examples, round_negatives], ignore_index=True
        ).drop_duplicates(["entity_id", "candidate_id"], keep="first")
        model = fit_final_pair_model(
            final_training_examples, selected_oof.model_type,
            args.seed + mining_round + 1, args.neural_top_k,
        )
        print(f"Final adversarial mining round {mining_round + 1} added "
              f"{len(round_negatives)} negatives; refit the pair model.")
    probability_calibrator = selected_oof.probability_calibrator
    calibration_report = output_dir / "calibration_drift.csv"
    heldout_pair_mask = selected_oof.pair_rows["fold"].to_numpy(dtype=int) == 0
    write_calibration_drift_report(
        calibration_report,
        selected_oof.pair_rows.loc[heldout_pair_mask, "probability"].to_numpy(dtype=float),
        selected_oof.crossfit_probabilities[
            selected_oof.pair_rows["fold"].to_numpy(dtype=int) != 0
        ],
        model,
        final_training_examples,
        probability_calibrator,
        args.seed,
    )
    print(f"Wrote OOF versus final-model score distribution report to {calibration_report}; "
          "final-model scores are in-sample diagnostics.")
    print(f"Final model trained on {len(final_source1):,} of {len(train_s1):,} labeled "
          "Source-1 entities.")
    margin_label = "disabled" if score_margin is None else f"{score_margin:.4f}"
    print(f"OOF-calibrated threshold={threshold:.6f}; within-entity score margin={margin_label}")

    del oof_results, selected_oof, sampled_training_examples, final_training_examples
    del train_candidate_diagnostics, train_index, train_lookup
    del train_s1, final_source1, oof_source1, truth, final_source1_ids, oof_source1_ids
    gc.collect()
    test_s1 = load_source(str(test_dir / "test_source1.tsv"))
    test_s2 = load_source(str(test_dir / "test_source2.tsv"))
    test_s3 = load_source(str(test_dir / "test_source3.tsv"))
    combined_test_sources = pd.concat([test_s2, test_s3], ignore_index=True, sort=False)
    del test_s2, test_s3
    print(f"Generating predictions for all {len(test_s1)} test Source-1 entities...")
    test_lookup = build_source_lookup(combined_test_sources)
    test_index = build_block_index(
        combined_test_sources,
        max_block_frequency=args.max_block_frequency,
        name_ngram_limit=args.name_ngram_limit,
        channel_limit_multiplier=args.candidate_channel_limit_multiplier,
        retrieval_context_mode=args.retrieval_context_mode,
        build_graph=not args.disable_target_graph,
        semantic_retrieval=args.semantic_retrieval,
        semantic_max_documents=args.semantic_index_max_targets,
        semantic_top_k=args.semantic_top_k,
        semantic_min_similarity=args.semantic_min_similarity,
    )
    del combined_test_sources
    predictions = predict_test_set(
        test_s1, None, model, threshold, score_margin,
        max_block_frequency=args.max_block_frequency,
        source_lookup=test_lookup,
        index=test_index,
        entity_decision=entity_decision,
        probability_calibrator=probability_calibrator,
    )
    root_output = Path("output")
    write_prediction_outputs(output_dir, root_output, predictions)
    print(f"Wrote outputs to {output_dir} and {root_output}")
    if args.validate_submission or args.check_submission_ids:
        validator = find_submission_validator(test_dir=test_dir)
        if validator is None:
            print("Warning: requested submission validation, but the challenge validator was not found; "
                  "prediction files are complete and remain available.")
        else:
            validation_command = [
                sys.executable, str(validator),
                "--matching", str(output_dir / "matching_results.tsv"),
                "--candidate", str(output_dir / "candidate_pairs.tsv"),
                "--test-dir", str(test_dir),
            ]
            if args.check_submission_ids:
                validation_command.append("--check-ids")
            validation_result = subprocess.run(validation_command, check=False)
            if validation_result.returncode != 0:
                raise SystemExit(validation_result.returncode)


if __name__ == "__main__":
    main()
