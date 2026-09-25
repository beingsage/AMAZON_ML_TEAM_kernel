#!/usr/bin/env python3
"""Business entity resolution pipeline for the Amazon ML Challenge 2026."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import re
import sys
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    "candidate_retrieval_rank_reciprocal",
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
    "name_postal_consistency", "phone_fragment_match", "candidate_source_s2", "candidate_source_s3",
    "has_name_both", "has_address_both",
]


class BlockIndex(defaultdict):
    """Posting lists plus target-corpus document frequencies for TF-IDF features."""

    def __init__(self) -> None:
        super().__init__(set)
        self.document_count = 0
        self.name_document_frequency: Counter[str] = Counter()
        self.address_document_frequency: Counter[str] = Counter()


@dataclass
class NeuralReranker:
    """Classical candidate ranker followed by a small MLP over pair features."""

    base_model: Any
    reranker: Any
    top_k: int


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
    threshold: float = 0.5
    score_margin: float | None = None
    cross_validation_f0_5: float = 0.0
    heldout_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EntityDecisionLayer:
    classifier: Any | None = None
    threshold: float = 0.5
    cardinality_classifier: Any | None = None


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
    "best_candidate_rank_reciprocal",
)
BLOCK_TYPE_NAMES = (
    "name_exact", "name_pair", "name_char_ngram", "name_token", "address_token",
    "postal", "house_number", "city_token",
)
CANDIDATE_EFFICIENCY_K = (50, 100, 200, 500, 1000)


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


def token_list(value: object) -> List[str]:
    text = normalize_text(value)
    return text.split() if text else []


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
    if len(all_ngrams) <= max_ngrams:
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
    name1_raw = normalize_text(_row_value(s1_row, "business_name"))
    name2_raw = normalize_text(_row_value(cand_row, "business_name"))
    name1_variants = name_variants(_row_value(s1_row, "business_name"))
    name2_variants = name_variants(_row_value(cand_row, "business_name"))
    name1 = core_name_tokens(_row_value(s1_row, "business_name"))
    name2 = core_name_tokens(_row_value(cand_row, "business_name"))
    name1_core, name2_core = " ".join(name1), " ".join(name2)

    address1_raw = normalize_text(_row_value(s1_row, "business_address"))
    address2_raw = normalize_text(_row_value(cand_row, "business_address"))
    address1 = address_tokens(_row_value(s1_row, "business_address"))
    address2 = address_tokens(_row_value(cand_row, "business_address"))
    country1 = normalize_country(_row_value(s1_row, "country"))
    country2 = normalize_country(_row_value(cand_row, "country"))
    city1 = address_city_tokens(_row_value(s1_row, "business_address"), country1)
    city2 = address_city_tokens(_row_value(cand_row, "business_address"), country2)
    state1 = address_state_tokens(_row_value(s1_row, "business_address"), country1)
    state2 = address_state_tokens(_row_value(cand_row, "business_address"), country2)
    nums1 = address_numbers(_row_value(s1_row, "business_address"))
    nums2 = address_numbers(_row_value(cand_row, "business_address"))
    units1 = unit_numbers(_row_value(s1_row, "business_address"))
    units2 = unit_numbers(_row_value(cand_row, "business_address"))
    phones1 = phone_fragments(_row_value(s1_row, "business_address"))
    phones2 = phone_fragments(_row_value(cand_row, "business_address"))
    postal1 = postal_codes(_row_value(s1_row, "business_address"))
    postal2 = postal_codes(_row_value(cand_row, "business_address"))
    house1 = house_number(_row_value(s1_row, "business_address"))
    house2 = house_number(_row_value(cand_row, "business_address"))
    landmarks1 = address_landmark_tokens(_row_value(s1_row, "business_address"))
    landmarks2 = address_landmark_tokens(_row_value(cand_row, "business_address"))

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
        "candidate_source_s2": float(candidate_id.startswith("S2-")),
        "candidate_source_s3": float(candidate_id.startswith("S3-")),
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


def predict_feature_rows(model: Any,
                         feature_rows: Sequence[Mapping[str, float]],
                         feature_names: Sequence[str] = FEATURE_NAMES,
                         batch_size: int = 50000) -> np.ndarray:
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


def _block_keys_for_values(name: object, address: object, country: object) -> Set[Tuple[str, str]]:
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
    for ngram in selected_name_ngrams(name):
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


def build_block_index(df: pd.DataFrame, max_block_frequency: int = 10000) -> BlockIndex:
    """Build a country-aware inverted index, optionally excluding very common keys."""
    counts: Counter[Tuple[str, str]] = Counter()
    index = BlockIndex()
    rows = df.itertuples(index=False)
    for row in rows:
        values = row._asdict()
        index.document_count += 1
        index.name_document_frequency.update({
            token for variant in name_variants(values.get("business_name", "")) for token in variant
        })
        index.address_document_frequency.update(set(address_tokens(values.get("business_address", ""))))
        counts.update(_block_keys_for_values(values.get("business_name", ""),
                                             values.get("business_address", ""),
                                             values.get("country", "")))

    for row in df.itertuples(index=False):
        values = row._asdict()
        entity_id = str(values.get("entity_id", ""))
        keys = _block_keys_for_values(values.get("business_name", ""),
                                      values.get("business_address", ""),
                                      values.get("country", ""))
        for key in keys:
            if max_block_frequency <= 0 or counts[key] <= max_block_frequency:
                index[key].add(entity_id)
    return index


def candidate_keys_for_record(row: Mapping[str, object] | pd.Series) -> Set[Tuple[str, str]]:
    # Word-token retrieval is scored separately with corpus IDF below.
    return {key for key in _block_keys_for_values(
        _row_value(row, "business_name"),
        _row_value(row, "business_address"),
        _row_value(row, "country"),
    ) if key[0] not in {"name_token", "address_token"}}


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
    candidate_ids: Set[str] = set()
    tfidf_scores: Dict[str, float] = defaultdict(float)
    for key in candidate_keys_for_record(s1_row):
        postings = index.get(key, set())
        candidate_ids.update(postings)
        if collect_blocks:
            block_candidates[key[0]].update(postings)
    if isinstance(index, BlockIndex):
        country = normalize_country(_row_value(s1_row, "country"))
        name_terms = {token for variant in name_variants(_row_value(s1_row, "business_name"))
                      for token in variant if len(token) >= 3 and token not in BLOCK_STOPWORDS}
        address_terms = {token for token in address_tokens(_row_value(s1_row, "business_address"))
                         if len(token) >= 4 and token not in BLOCK_STOPWORDS}
        if collect_blocks:
            block_candidates.setdefault("name_token", set())
            block_candidates.setdefault("address_token", set())

        def retrieve_tfidf(kind: str, terms: Set[str], frequencies: Mapping[str, int]) -> None:
            for token in terms:
                df = frequencies.get(token, 0)
                idf = math.log((1.0 + index.document_count) / (1.0 + df)) + 1.0
                for scope in (country, "*"):
                    postings = index.get((kind, f"{scope}|{token}"), set())
                    candidate_ids.update(postings)
                    if collect_blocks:
                        block_candidates[kind].update(postings)
                    for candidate_id in postings:
                        tfidf_scores[candidate_id] += idf

        retrieve_tfidf("name_token", name_terms, index.name_document_frequency)
        retrieve_tfidf("address_token", address_terms, index.address_document_frequency)
    valid_ids = lookup.index if isinstance(lookup, pd.DataFrame) else lookup
    if collect_blocks:
        for block_type in list(block_candidates):
            block_candidates[block_type] = {
                eid for eid in block_candidates[block_type]
                if eid.startswith(("S2-", "S3-")) and eid in valid_ids
            }
    candidate_ids = {eid for eid in candidate_ids
                     if eid.startswith(("S2-", "S3-")) and eid in valid_ids}
    ordered = sorted(candidate_ids, key=lambda eid: (-tfidf_scores.get(eid, 0.0), eid))
    return ordered, dict(block_candidates) if collect_blocks else {}


def generate_candidates(s1_row: Mapping[str, object] | pd.Series,
                        index: Dict[Tuple[str, str], Set[str]],
                        lookup: Mapping[str, object] | pd.DataFrame) -> List[str]:
    return generate_candidate_details(s1_row, index, lookup, collect_blocks=False)[0]


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
                    features = build_pair_features(
                        s1_values, lookup_record(source_lookup, candidate_id), index,
                        candidate_rank=candidate_ranks[candidate_id],
                    )
                    ranked.append((negative_hardness(features), candidate_id, negative_category(features)))
                ranked.sort(key=lambda item: (-item[0], item[1]))
                hard_count = min(target_negatives, max(1, int(np.ceil(target_negatives * 0.8))))
                category_order = ("same_name_wrong_address", "same_address_wrong_name",
                                  "same_city", "same_country", "other")
                chosen: List[Tuple[float, str, str]] = []
                chosen_ids: Set[str] = set()
                category_quota = max(1, hard_count // len(category_order))
                for category in category_order:
                    category_rows = [item for item in ranked if item[2] == category]
                    for item in category_rows[:category_quota]:
                        chosen.append(item)
                        chosen_ids.add(item[1])
                for item in ranked:
                    if len(chosen) >= hard_count:
                        break
                    if item[1] not in chosen_ids:
                        chosen.append(item)
                        chosen_ids.add(item[1])
                remaining = [item for item in ranked if item[1] not in chosen_ids]
                random_count = min(len(remaining), min(random_negatives, target_negatives - len(chosen)))
                if random_count:
                    chosen_indices = sorted(rng.choice(len(remaining), size=random_count, replace=False).tolist())
                    chosen.extend(remaining[i] for i in chosen_indices)
                selected_negative_ids = [candidate_id for _, candidate_id, _ in chosen]
            else:
                selected_positions = rng.choice(len(negative_ids), size=target_negatives, replace=False)
                selected_negative_ids = [negative_ids[position] for position in sorted(selected_positions)]
            negative_features = [
                (candidate_id, build_pair_features(
                    s1_values, lookup_record(source_lookup, candidate_id), index,
                    candidate_rank=candidate_ranks[candidate_id],
                ))
                for candidate_id in selected_negative_ids
            ]
        else:
            negative_features = [
                (candidate_id, build_pair_features(
                    s1_values, lookup_record(source_lookup, candidate_id), index,
                    candidate_rank=candidate_ranks[candidate_id],
                ))
                for candidate_id in negative_ids
            ]

        for candidate_id in sorted(positive_ids):
            examples.append({
                "entity_id": entity_id,
                "candidate_id": candidate_id,
                "label": 1,
                "features": build_pair_features(
                    s1_values, lookup_record(source_lookup, candidate_id), index,
                    candidate_rank=candidate_ranks[candidate_id],
                ),
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
    ids = set(entity_ids) if entity_ids is not None else set(predicted_matches) | set(truth)
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
            unique_scores, starts, counts = np.unique(
                eligible_scores, return_index=True, return_counts=True
            )
            true_counts_by_score = np.add.reduceat(eligible_labels.astype(np.int64), starts)
            for candidate_score, removed_count, removed_true_count in zip(
                unique_scores, counts, true_counts_by_score
            ):
                next_score = entity_score(
                    true_positive_count - int(removed_true_count),
                    predicted_count - int(removed_count),
                )
                delta = next_score - current_score
                if delta:
                    event_thresholds.append(float(np.nextafter(candidate_score, np.inf)))
                    event_deltas.append(delta)
                true_positive_count -= int(removed_true_count)
                predicted_count -= int(removed_count)
                current_score = next_score

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
            if score > best_score or more_conservative_tie:
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
    best_name_position = int(np.argmax(name_scores))
    best_address_position = int(np.argmax(address_scores))
    return [
        float(probabilities[order[0]]),
        second_probability,
        float(probabilities[order[0]] - second_probability),
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
    best_score = total_score
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


def cardinality_adjusted_predictions(pair_df: pd.DataFrame,
                                     probabilities: np.ndarray,
                                     threshold: float,
                                     score_margin: float | None,
                                     entity_ids: Sequence[str],
                                     predicted_cardinality: np.ndarray | None) -> Dict[str, Set[str]]:
    predictions = predictions_from_pair_scores(
        pair_df, probabilities, threshold, score_margin, entity_ids
    )
    if predicted_cardinality is None:
        return predictions
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
                if probabilities[best_position] >= threshold else set()
            )
            continue
        current = predictions.get(entity_id, set())
        if len(current) < 2:
            second_position = ranked[1] if len(ranked) > 1 else None
            if second_position is not None:
                best_probability = float(probabilities[ranked[0]])
                second_probability = float(probabilities[second_position])
                within_margin = (score_margin is None or
                                 second_probability >= best_probability - score_margin)
                if second_probability >= max(0.0, threshold - 0.1) and within_margin:
                    current.update({
                        str(pair_df.iloc[ranked[0]]["candidate_id"]),
                        str(pair_df.iloc[second_position]["candidate_id"]),
                    })
        predictions[entity_id] = current
    return predictions


def train_oof_entity_decision_layers(result: OOFModelResult,
                                     truth: Mapping[str, Set[str]],
                                     folds: int,
                                     threshold: float,
                                     score_margin: float | None) -> Tuple[EntityDecisionLayer | None,
                                                                          float, float, float]:
    """Fit gate/cardinality layers on OOF entities and evaluate only fold zero."""
    pair_rows = result.pair_rows
    entity_ids = result.entity_ids
    fold_by_entity = result.fold_by_entity
    if folds < 3 or not len(entity_ids):
        return None, 0.0, 0.0, 0.0

    tune_fold = folds - 1
    fit_mask = (fold_by_entity > 0) & (fold_by_entity < tune_fold)
    tune_mask = fold_by_entity == tune_fold
    eval_mask = fold_by_entity == 0
    linked_labels = np.asarray([bool(truth.get(entity_id, set())) for entity_id in entity_ids], dtype=int)
    cardinality_labels = np.asarray([min(2, len(truth.get(entity_id, set())))
                                     for entity_id in entity_ids], dtype=int)
    entity_features = result.entity_features.copy()
    entity_pos = {entity_id: position for position, entity_id in enumerate(entity_ids)}
    pair_positions = pair_rows["entity_id"].astype(str).map(entity_pos).to_numpy(dtype=np.int64)
    above_threshold = np.bincount(
        pair_positions,
        weights=(pair_rows["probability"].to_numpy(dtype=float) >= threshold).astype(int),
        minlength=len(entity_ids),
    )
    entity_features[:, ENTITY_FEATURE_NAMES.index("number_above_pair_threshold")] = above_threshold

    fit_positions = np.flatnonzero(fit_mask)
    tune_positions = np.flatnonzero(tune_mask)
    eval_positions = np.flatnonzero(eval_mask)
    if len(fit_positions) < 20 or len(tune_positions) == 0 or len(eval_positions) == 0:
        return None, 0.0, 0.0, 0.0

    calibration_pair_rows = pair_rows[pair_rows["fold"].astype(int) == 0]
    # A compact fold-level selection function keeps the evaluation partition
    # completely separate from gate fitting and threshold tuning.
    pair_predictions_by_fold: Dict[int, Dict[str, Set[str]]] = {}
    for fold in range(folds):
        fold_rows = pair_rows[pair_rows["fold"].astype(int) == fold]
        fold_ids = [entity_id for position, entity_id in enumerate(entity_ids)
                    if fold_by_entity[position] == fold]
        pair_predictions_by_fold[fold] = predictions_from_pair_scores(
            fold_rows, fold_rows["probability"].to_numpy(dtype=float),
            threshold, score_margin, fold_ids,
        )
    _ = calibration_pair_rows

    fit_linked = linked_labels[fit_positions]
    gate_classifier = None
    gate_threshold = 1.1
    tune_ids = [entity_ids[position] for position in tune_positions]
    eval_ids = [entity_ids[position] for position in eval_positions]
    tune_predictions = pair_predictions_by_fold[tune_fold]
    eval_predictions = pair_predictions_by_fold[0]
    if len(np.unique(fit_linked)) == 2 and np.bincount(fit_linked, minlength=2).min() >= 10:
        gate_classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0),
        )
        gate_classifier.fit(entity_features[fit_positions], fit_linked)
        tune_gate_probabilities = gate_classifier.predict_proba(entity_features[tune_positions])[:, 1]
        gate_threshold = best_gate_threshold(
            tune_gate_probabilities, tune_predictions, truth, tune_ids
        )

    holdout_rows = pair_rows[pair_rows["fold"].astype(int) == 0]
    holdout_probabilities = holdout_rows["probability"].to_numpy(dtype=float)
    baseline_score = score_entity_predictions(eval_predictions, truth, eval_ids)
    gate_score = baseline_score
    accepted_eval: Dict[str, Set[str]] = eval_predictions
    if gate_classifier is not None:
        eval_gate_probabilities = gate_classifier.predict_proba(entity_features[eval_positions])[:, 1]
        accepted_eval = {
            entity_id: eval_predictions.get(entity_id, set()) if probability >= gate_threshold else set()
            for entity_id, probability in zip(eval_ids, eval_gate_probabilities)
        }
        gate_score = score_entity_predictions(accepted_eval, truth, eval_ids)
        if gate_score <= baseline_score:
            gate_classifier = None

    cardinality_classifier = None
    cardinality_score = gate_score
    fit_cardinality = cardinality_labels[fit_positions]
    class_counts = np.bincount(fit_cardinality, minlength=3)
    if np.count_nonzero(class_counts >= 10) >= 2:
        cardinality_classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=1),
        )
        cardinality_classifier.fit(entity_features[fit_positions], fit_cardinality)
        eval_cardinality = cardinality_classifier.predict(entity_features[eval_positions])
        gated_pair_rows = holdout_rows
        cardinality_predictions = cardinality_adjusted_predictions(
            gated_pair_rows, holdout_probabilities, threshold, score_margin, eval_ids,
            eval_cardinality,
        )
        if gate_classifier is not None:
            eval_gate_probabilities = gate_classifier.predict_proba(entity_features[eval_positions])[:, 1]
            cardinality_predictions = {
                entity_id: cardinality_predictions.get(entity_id, set())
                if probability >= gate_threshold else set()
                for entity_id, probability in zip(eval_ids, eval_gate_probabilities)
            }
        cardinality_score = score_entity_predictions(cardinality_predictions, truth, eval_ids)
        if cardinality_score <= gate_score:
            cardinality_classifier = None

    if gate_classifier is None and cardinality_classifier is None:
        return None, baseline_score, gate_score, cardinality_score
    return EntityDecisionLayer(
        classifier=gate_classifier,
        threshold=gate_threshold,
        cardinality_classifier=cardinality_classifier,
    ), baseline_score, gate_score, cardinality_score


def feature_matrix(pair_df: pd.DataFrame,
                   feature_names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    if pair_df.empty:
        return np.empty((0, len(feature_names)), dtype=float)
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
    val_top = select_top_k_pairs(val_df, base_model, top_k)
    threshold, score_margin, validation_score = threshold_search(
        classifier, val_top, truth, validation_entity_ids
    )
    return NeuralReranker(base_model=base_model, reranker=classifier, top_k=top_k), threshold, score_margin, validation_score


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
    X_train = np.asarray([features_to_vector(value, feature_names) for value in train_df["features"]], dtype=float)
    y_train = train_df["label"].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training candidates contain only one class; improve blocking or sampling.")

    model = make_pair_estimator(model_type, seed)
    model.fit(X_train, y_train)
    threshold, score_margin, validation_score = threshold_search(
        model, val_df, truth, validation_entity_ids, feature_names
    )
    return model, threshold, score_margin, validation_score


def make_pair_estimator(model_type: str, seed: int) -> Any:
    if model_type == "logistic":
        return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    if model_type == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is not installed. Install requirements.txt to use --model lightgbm.")
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Unsupported model: {model_type}")


def run_entity_oof_validation(source1: pd.DataFrame,
                              truth: Mapping[str, Set[str]],
                              source_lookup: pd.DataFrame,
                              index: BlockIndex,
                              model_types: Sequence[str],
                              folds: int,
                              seed: int,
                              negatives_per_positive: int,
                              random_negatives: int,
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
    fold_recalls: Dict[int, float] = {}
    holdout_examples = pd.DataFrame()
    holdout_source1 = pd.DataFrame()

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
        fold_recalls[fold] = candidate_recall_value
        if fold == 0:
            holdout_examples = validation_examples
            holdout_source1 = validation_source1.copy()
        print_candidate_diagnostics(f"OOF fold {fold + 1}/{folds}", diagnostics, candidate_recall_value)
        y_train = fold_training_examples["label"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"OOF fold {fold + 1} training pairs contain only one class.")
        X_train = feature_matrix(fold_training_examples)

        for model_type in model_types:
            model = make_pair_estimator(model_type, seed + fold)
            model.fit(X_train, y_train)
            probabilities = predict_pair_probabilities(model, validation_examples)
            fold_rows = validation_examples[["entity_id", "candidate_id", "label"]].copy()
            fold_rows["probability"] = probabilities
            fold_rows["fold"] = fold
            oof_rows[model_type].append(fold_rows)

            gate_features = entity_decision_feature_matrix(
                validation_examples, probabilities, validation_ids, pair_threshold=0.0
            )
            positions = [entity_position[entity_id] for entity_id in validation_ids]
            oof_entity_features[model_type][positions] = gate_features

            if adversarial_negatives_per_entity > 0 and not validation_examples.empty:
                for entity_id, group in validation_examples.groupby("entity_id", sort=False):
                    negative_positions = np.flatnonzero(group["label"].astype(int).to_numpy() == 0)
                    if not len(negative_positions):
                        continue
                    global_positions = group.index.to_numpy()[negative_positions]
                    ranked = sorted(
                        global_positions,
                        key=lambda position: (-float(probabilities[validation_examples.index.get_loc(position)]),
                                              str(validation_examples.at[position, "candidate_id"])),
                    )
                    selected_positions = ranked[:adversarial_negatives_per_entity]
                    if selected_positions:
                        mined_rows[model_type].append(validation_examples.loc[selected_positions].copy())
            del model

        del X_train, y_train, fold_training_examples, validation_examples
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
        )
    return results, all_training_examples, train_candidate_recall, train_diagnostics


def calibrate_and_compare_oof_models(results: Mapping[str, OOFModelResult],
                                     truth: Mapping[str, Set[str]],
                                     folds: int) -> OOFModelResult:
    for result in results.values():
        pair_rows = result.pair_rows
        fold_values = pair_rows["fold"].to_numpy(dtype=int)
        fold_scores: List[float] = []
        # Fold 0 remains an independent final check. Model-family comparison
        # uses only folds 1..N-1 and thresholds calibrated on the other folds.
        for evaluation_fold in range(1, folds):
            calibration_mask = (fold_values != 0) & (fold_values != evaluation_fold)
            evaluation_mask = fold_values == evaluation_fold
            calibration_rows = pair_rows.loc[calibration_mask]
            evaluation_rows = pair_rows.loc[evaluation_mask]
            calibration_ids = [
                entity_id for position, entity_id in enumerate(result.entity_ids)
                if result.fold_by_entity[position] not in {0, evaluation_fold}
            ]
            evaluation_ids = [
                entity_id for position, entity_id in enumerate(result.entity_ids)
                if result.fold_by_entity[position] == evaluation_fold
            ]
            threshold, margin, _ = threshold_search_from_scores(
                calibration_rows, calibration_rows["probability"].to_numpy(dtype=float),
                truth, calibration_ids,
            )
            candidate_recall_value = candidate_recall_for_rows(evaluation_rows, truth, evaluation_ids)
            metrics = evaluate_scored_pairs(
                evaluation_rows, evaluation_rows["probability"].to_numpy(dtype=float),
                truth, evaluation_ids, threshold, margin, candidate_recall_value,
            )
            fold_scores.append(metrics["entity_f0_5"])
        result.cross_validation_f0_5 = float(np.mean(fold_scores)) if fold_scores else 0.0

        calibration_mask = fold_values != 0
        calibration_rows = pair_rows.loc[calibration_mask]
        calibration_ids = [
            entity_id for position, entity_id in enumerate(result.entity_ids)
            if result.fold_by_entity[position] != 0
        ]
        result.threshold, result.score_margin, _ = threshold_search_from_scores(
            calibration_rows, calibration_rows["probability"].to_numpy(dtype=float),
            truth, calibration_ids,
        )
        holdout_mask = fold_values == 0
        holdout_rows = pair_rows.loc[holdout_mask]
        holdout_ids = [
            entity_id for position, entity_id in enumerate(result.entity_ids)
            if result.fold_by_entity[position] == 0
        ]
        result.heldout_metrics = evaluate_scored_pairs(
            holdout_rows, holdout_rows["probability"].to_numpy(dtype=float),
            truth, holdout_ids, result.threshold, result.score_margin,
            candidate_recall_for_rows(holdout_rows, truth, holdout_ids),
        )
        print(f"OOF {result.model_type}: CV entity F0.5={result.cross_validation_f0_5:.4f}; "
              f"independent fold-1 entity F0.5={result.heldout_metrics['entity_f0_5']:.4f}; "
              f"candidate recall={result.heldout_metrics['candidate_pair_recall']:.4f}")
    return max(results.values(), key=lambda result: result.cross_validation_f0_5)


def candidate_recall_for_rows(pair_rows: pd.DataFrame,
                              truth: Mapping[str, Set[str]],
                              entity_ids: Sequence[str]) -> float:
    total_truth = sum(len(truth.get(str(entity_id), set())) for entity_id in entity_ids)
    found_truth = int(pair_rows["label"].astype(int).sum()) if not pair_rows.empty else 0
    return found_truth / total_truth if total_truth else 1.0


def feature_ablation_sets() -> Dict[str, List[str]]:
    address_prefixes = ("address_", "common_address", "number_", "house_number", "unit_number",
                        "postal_code", "city_", "state_", "tail_component", "phone_fragment",
                        "address_landmark")
    location_prefixes = ("number_", "house_number", "unit_number", "postal_code", "city_",
                         "state_", "tail_component", "same_country", "country_conflict",
                         "phone_fragment", "address_landmark")
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


def predict_test_set(test_source1: pd.DataFrame,
                     source2_source3_df: pd.DataFrame | None,
                     model: Any,
                     threshold: float,
                     score_margin: float | None = None,
                     max_block_frequency: int = 10000,
                     source_lookup: Mapping[str, object] | pd.DataFrame | None = None,
                     index: Dict[Tuple[str, str], Set[str]] | None = None,
                     entity_decision: EntityDecisionLayer | None = None,
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
        candidates = generate_candidates(s1_values, index, source_lookup)
        if not candidates:
            yield s1_id, candidates, []
            continue

        feature_rows = [build_pair_features(
            s1_values, lookup_record(source_lookup, candidate_id), index,
            candidate_rank=position + 1,
        ) for position, candidate_id in enumerate(candidates)]
        first_stage_model = model.base_model if isinstance(model, NeuralReranker) else model
        base_probabilities = None
        if entity_decision is not None or isinstance(model, NeuralReranker):
            base_probabilities = predict_feature_rows(first_stage_model, feature_rows)
        if entity_decision is not None:
            entity_features = entity_decision_feature_vector(feature_rows, base_probabilities, threshold)
            linked_probability = entity_decision.classifier.predict_proba([entity_features])[0, 1]
            if linked_probability < entity_decision.threshold:
                yield s1_id, candidates, []
                continue
        if isinstance(model, NeuralReranker):
            top_positions = sorted(
                range(len(candidates)),
                key=lambda position: (-base_probabilities[position], candidates[position]),
            )[:model.top_k]
            reranked = predict_feature_rows(model.reranker, [feature_rows[position] for position in top_positions])
            best_probability = float(reranked.max())
            selected = [candidates[position] for position, probability in zip(top_positions, reranked)
                        if probability >= threshold and (score_margin is None or
                                                        probability >= best_probability - score_margin)]
            yield s1_id, candidates, sorted(selected)
            continue

        probabilities = (base_probabilities if base_probabilities is not None
                         else predict_feature_rows(model, feature_rows))
        best_probability = float(probabilities.max())
        selected = [candidate_id for candidate_id, probability in zip(candidates, probabilities)
                    if probability >= threshold and (score_margin is None or
                                                      probability >= best_probability - score_margin)]
        # Candidate IDs are sorted, which keeps serialized match lists deterministic.
        yield s1_id, candidates, selected


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Business entity resolution pipeline")
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", default="student_resource/output")
    parser.add_argument("--sample-train-rows", type=int, default=50000,
                        help="Stratified Source-1 training sample; use 0 for every training entity.")
    parser.add_argument("--sample-validation-rows", type=int, default=20000,
                        help="Stratified entity-level validation sample; use 0 for the full holdout.")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--negatives-per-positive", type=int, default=20)
    parser.add_argument("--random-negatives", type=int, default=5)
    parser.add_argument("--model", choices=("logistic", "lightgbm", "compare"), default="logistic",
                        help="Pair matcher, or compare both model families on the validation sample.")
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
                        help="Ignore more common posting lists; use 0 to retain every block for maximum recall.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_s1 = load_source(str(train_dir / "train_source1.tsv"))
    train_s2 = load_source(str(train_dir / "train_source2.tsv"))
    train_s3 = load_source(str(train_dir / "train_source3.tsv"))
    truth = parse_truth(str(train_dir / "train_ground_truth.tsv"))
    combined_train_sources = pd.concat([train_s2, train_s3], ignore_index=True, sort=False)

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
        write_normalization_audit(
            audit_path, train_s1, args.normalization_audit_rows, args.seed
        )
        print(f"Wrote normalization review sample to {audit_path}")

    train_pool, validation_pool = split_training_data(
        train_s1, validation_fraction=args.validation_fraction, random_state=args.seed
    )
    if args.neural_top_k < 1:
        parser.error("--neural-top-k must be at least 1")
    full_name_frequencies: Counter[str] = Counter()
    for value in train_s1["business_name"]:
        name = " ".join(core_name_tokens(value))
        if name:
            full_name_frequencies[name] += 1
    train_pool = stratified_sample_source1(
        train_pool, truth, args.sample_train_rows, full_name_frequencies, args.seed
    )
    validation_pool = stratified_sample_source1(
        validation_pool, truth, args.sample_validation_rows, full_name_frequencies, args.seed + 1
    )
    validation_entity_ids = validation_pool["entity_id"].astype(str).tolist()
    print(f"Train Source1 sample: {len(train_pool)}; validation Source1 sample: {len(validation_pool)}")

    print("Building training-source lookup and high-recall block index...")
    train_lookup = build_source_lookup(combined_train_sources)
    train_index = build_block_index(combined_train_sources, max_block_frequency=args.max_block_frequency)
    del train_s2, train_s3, combined_train_sources
    train_examples, train_candidate_recall, train_candidate_diagnostics = build_training_examples(
        train_pool, None, truth,
        negatives_per_positive=args.negatives_per_positive,
        random_negatives=args.random_negatives,
        training=True,
        seed=args.seed,
        source_lookup=train_lookup,
        index=train_index,
    )
    val_examples, val_candidate_recall, val_candidate_diagnostics = build_training_examples(
        validation_pool, None, truth,
        training=False,
        seed=args.seed,
        source_lookup=train_lookup,
        index=train_index,
    )
    print(f"Training pairs: {len(train_examples)}; validation pairs: {len(val_examples)}")
    print_candidate_diagnostics("Training", train_candidate_diagnostics, train_candidate_recall)
    print_candidate_diagnostics("Validation", val_candidate_diagnostics, val_candidate_recall)

    model, threshold, score_margin, validation_score = train_model(
        train_examples, val_examples, truth, validation_entity_ids,
        seed=args.seed, model_type=args.model,
    )
    if args.neural_reranker:
        base_result = (model, threshold, score_margin, validation_score)
        reranker_result = train_neural_reranker(
            train_examples, val_examples, model, args.neural_top_k,
            truth, validation_entity_ids, args.seed,
        )
        print(f"Optional MLP reranker validation entity-macro F0.5: {reranker_result[3]:.4f} "
              f"(top K={args.neural_top_k}); first-stage score: {base_result[3]:.4f}")
        if reranker_result[3] > base_result[3]:
            model, threshold, score_margin, validation_score = reranker_result
        else:
            print("Keeping the first-stage matcher because it scored at least as well on validation.")
    entity_decision, entity_base_score, entity_gate_score = train_entity_decision_layer(
        model, val_examples, truth, validation_entity_ids, threshold, score_margin, args.seed
    )
    print(f"Entity decision held-out F0.5: base={entity_base_score:.4f}, "
          f"gated={entity_gate_score:.4f}")
    if entity_decision is None:
        print("Entity decision layer was not selected on held-out validation entities.")
    else:
        print(f"Entity decision layer enabled (gate threshold={entity_decision.threshold:.6f}).")
    if args.run_ablations:
        random_negative_examples, _, _ = build_training_examples(
            train_pool, None, truth,
            negatives_per_positive=args.negatives_per_positive,
            random_negatives=args.random_negatives,
            training=True,
            mine_hard_negatives=False,
            seed=args.seed,
            source_lookup=train_lookup,
            index=train_index,
        )
        if args.model == "compare":
            ablation_model_types = ["logistic", "lightgbm"]
        else:
            ablation_model_types = [args.model]
        report_path = output_dir / "ablation_results.csv"
        run_ablation_suite(
            train_examples, random_negative_examples, val_examples, truth,
            validation_entity_ids, ablation_model_types, args.seed,
            val_candidate_recall, report_path,
            train_pool, validation_pool, train_lookup.reset_index(), train_lookup,
            args.negatives_per_positive, args.random_negatives, args.max_block_frequency,
        )
        del random_negative_examples
        print(f"Wrote validation ablation results to {report_path}")
    margin_label = "disabled" if score_margin is None else f"{score_margin:.2f}"
    print(f"Best entity-macro F0.5 threshold: {threshold:.6f}; rank margin: {margin_label} (score={validation_score:.4f})")

    # Training-side source indexes and sampled pair features are no longer needed.
    del train_lookup, train_index, train_examples, val_examples
    del train_candidate_diagnostics, val_candidate_diagnostics
    del train_pool, validation_pool, train_s1, truth
    gc.collect()

    test_s1 = load_source(str(test_dir / "test_source1.tsv"))
    test_s2 = load_source(str(test_dir / "test_source2.tsv"))
    test_s3 = load_source(str(test_dir / "test_source3.tsv"))
    combined_test_sources = pd.concat([test_s2, test_s3], ignore_index=True, sort=False)
    del test_s2, test_s3
    print(f"Generating predictions for all {len(test_s1)} test Source-1 entities...")
    test_lookup = build_source_lookup(combined_test_sources)
    test_index = build_block_index(combined_test_sources, max_block_frequency=args.max_block_frequency)
    del combined_test_sources
    predictions = predict_test_set(
        test_s1, None, model, threshold, score_margin,
        max_block_frequency=args.max_block_frequency,
        source_lookup=test_lookup,
        index=test_index,
        entity_decision=entity_decision,
    )
    root_output = Path("output")
    write_prediction_outputs(output_dir, root_output, predictions)
    print(f"Wrote outputs to {output_dir} and {root_output}")


if __name__ == "__main__":
    main()
