"""A small Python implementation of the architecture shown in the diagram.

The pipeline is organized into four main stages:
- input text -> tokenization -> words
- modeling module
- error detection module
- user correction module

The implementation is intentionally lightweight but functional. It can be used
as a starting point for expanding language resources, stemming rules, and
candidate ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from html import unescape
from collections import Counter, defaultdict
import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from urllib.error import URLError
from urllib.request import urlopen


WORD_RE = re.compile(r"[A-Za-z0-9']+(?:-[A-Za-z0-9']+)*")
TAGALOG_LINE_RE = re.compile(r"^T:\s*(.+)$")

ENGLISH_WORDS_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
TAGALOG_SOURCE_URL = "https://raymelon.github.io/tagalog-dictionary-scraper/tagalog_dict.txt"
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)
ENGLISH_CACHE_PATH = CACHE_DIR / "english_words.txt"
TAGALOG_CACHE_PATH = CACHE_DIR / "tagalog_words.txt"


def tokenize(text: str) -> List[str]:
	return WORD_RE.findall(text.lower())


def normalize_spaces(text: str) -> str:
	return re.sub(r"\s+", " ", text).strip()


def levenshtein_distance(left: str, right: str) -> int:
	if left == right:
		return 0
	if not left:
		return len(right)
	if not right:
		return len(left)

	previous_row = list(range(len(right) + 1))
	for i, left_char in enumerate(left, start=1):
		current_row = [i]
		for j, right_char in enumerate(right, start=1):
			insertion = current_row[j - 1] + 1
			deletion = previous_row[j] + 1
			substitution = previous_row[j - 1] + (left_char != right_char)
			current_row.append(min(insertion, deletion, substitution))
		previous_row = current_row
	return previous_row[-1]


def ngrams(tokens: Sequence[str], size: int) -> List[Tuple[str, ...]]:
	if size <= 0:
		return []
	return [tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)]


def simple_stem(word: str) -> str:
	return filipino_stem(word)


def character_trigrams(text: str) -> List[str]:
	if len(text) < 3:
		return [text] if text else []
	return [text[index : index + 3] for index in range(len(text) - 2)]


def dice_coefficient(left: str, right: str) -> float:
	left_bigrams = [left[index : index + 2] for index in range(len(left) - 1)] or [left]
	right_bigrams = [right[index : index + 2] for index in range(len(right) - 1)] or [right]
	left_counts = Counter(left_bigrams)
	right_counts = Counter(right_bigrams)
	overlap = sum(min(left_counts[gram], right_counts[gram]) for gram in left_counts)
	return (2.0 * overlap) / (len(left_bigrams) + len(right_bigrams))


def filipino_phonetic_code(word: str) -> str:
	"""Conservative Filipino/Taglish phonetic normalization.

	Goal: help rank *plausible* orthographic variants without collapsing many
	different words into the same code.
	"""
	code = word.lower()

	# Keep only very common spelling->sound variants.
	for source, target in (
		("ph", "f"),
		("qu", "k"),
		("cw", "k"),
		("x", "ks"),
		("v", "b"),
		("z", "s"),
	):
		code = code.replace(source, target)

	# Keep ending normalization minimal: only normalize explicit "tion".
	code = re.sub(r"tion$", "syon", code)

	# Remove silent/inserted 'h' when it precedes a consonant.
	code = re.sub(r"h(?=[bcdfghjklmnpqrstvwxyz])", "", code)

	# Collapse repeated letters, but do it conservatively.
	# (Avoid turning "mm"/"nn"-type legitimate patterns into too-short codes.)
	code = re.sub(r"(.)\1{2,}", r"\1\1", code)
	return code




def filipino_stem(word: str) -> str:
	"""Conservative stemming.

	The current aggressive affix stripping can destroy correct Tagalog stems,
	causing the checker to propose wrong words.

	We only normalize the token minimally here; longer/true morphology should
	come from better rules + lexicon coverage.
	"""
	stem = word.lower().replace("-", "")
	if len(stem) <= 4:
		return stem

	# Very small, safe suffix handling (do not remove if it would be too
	# destructive).
	# Keep only common English-like endings + plural 's'.
	for suffix in ("s", "ed", "ing", "ly"):
		if len(stem) >= len(suffix) + 4 and stem.endswith(suffix):
			stem = stem[: -len(suffix)]
			break

	return stem



def fetch_remote_text(url: str) -> str:
	with urlopen(url, timeout=30) as response:
		content = response.read()
		for encoding in ("utf-8", "latin-1"):
			try:
				return content.decode(encoding)
			except UnicodeDecodeError:
				continue
		return content.decode("utf-8", errors="replace")


def read_cached_text(path: Path) -> str:
	if not path.exists():
		return ""
	return path.read_text(encoding="utf-8", errors="replace")


def write_cached_text(path: Path, text: str) -> None:
	path.write_text(text, encoding="utf-8")


def extract_word_set(text: str) -> Set[str]:
	words: Set[str] = set()
	for token in WORD_RE.findall(unescape(text).lower()):
		if len(token) > 1 or token.isalpha():
			words.add(token)
	return words


@dataclass
class AutomatonNode:
	transitions: Dict[str, int] = field(default_factory=dict)
	terminal: bool = False


class DeterministicAutomaton:
	def __init__(self, words: Iterable[str] | None = None) -> None:
		self.nodes: List[AutomatonNode] = [AutomatonNode()]
		self.word_set: Set[str] = set()
		if words is not None:
			self.build(words)

	def build(self, words: Iterable[str]) -> None:
		self.nodes = [AutomatonNode()]
		self.word_set = {word.strip().lower() for word in words if word and word.strip()}
		for word in sorted(self.word_set):
			node_index = 0
			for character in word:
				next_index = self.nodes[node_index].transitions.get(character)
				if next_index is None:
					next_index = len(self.nodes)
					self.nodes.append(AutomatonNode())
					self.nodes[node_index].transitions[character] = next_index
				node_index = next_index
			self.nodes[node_index].terminal = True

	def contains(self, word: str) -> bool:
		node_index = 0
		for character in word.lower():
			next_index = self.nodes[node_index].transitions.get(character)
			if next_index is None:
				return False
			node_index = next_index
		return self.nodes[node_index].terminal

	def iter_words(self) -> Set[str]:
		return set(self.word_set)


def load_english_words() -> Set[str]:
	cached_text = read_cached_text(ENGLISH_CACHE_PATH)
	try:
		text = fetch_remote_text(ENGLISH_WORDS_URL)
		write_cached_text(ENGLISH_CACHE_PATH, text)
		return {line.strip().lower() for line in text.splitlines() if line.strip()}
	except (URLError, TimeoutError, OSError):
		if cached_text:
			return {line.strip().lower() for line in cached_text.splitlines() if line.strip()}
		return {
			"the",
			"this",
			"is",
			"we",
			"are",
			"and",
			"hello",
			"world",
			"computer",
			"language",
			"model",
			"system",
			"working",
			"work",
			"error",
			"suggestion",
			"text",
			"token",
			"processing",
			"check",
		}


def load_tagalog_words() -> Set[str]:
	cached_text = read_cached_text(TAGALOG_CACHE_PATH)
	try:
		text = fetch_remote_text(TAGALOG_SOURCE_URL)
		write_cached_text(TAGALOG_CACHE_PATH, text)
	except (URLError, TimeoutError, OSError):
		if cached_text:
			text = cached_text
		else:
			return {
				"ang",
				"mga",
				"kumusta",
				"kamusta",
				"salamat",
				"ako",
				"ikaw",
				"ito",
				"yan",
				"oras",
				"sistema",
				"teksto",
				"tama",
			}

	# The raymelon tagalog-dictionary-scraper provides a newline-delimited word list.
	# Each line is expected to be a single Tagalog word (often already lowercase).
	words: Set[str] = set()
	for raw_line in text.splitlines():
		line = raw_line.strip()
		if not line:
			continue

		# Backward compatibility: if the old dataset format is ever used, keep parsing it.
		match = TAGALOG_LINE_RE.match(line)
		if match:
			entry = unescape(match.group(1)).lower()
		else:
			entry = unescape(line).lower()

		for token in WORD_RE.findall(entry):
			if len(token) > 1 or token.isalpha():
				words.add(token)
	return words


@dataclass
class ModelingModule:
	english_word_list: Set[str]
	tagalog_word_list: Set[str]
	ngram_size: int = 3
	ngram_threshold: float = 0.10
	english_automaton: DeterministicAutomaton = field(default_factory=DeterministicAutomaton)
	tagalog_automaton: DeterministicAutomaton = field(default_factory=DeterministicAutomaton)
	stemmed_english_automaton: DeterministicAutomaton = field(default_factory=DeterministicAutomaton)
	stemmed_tagalog_automaton: DeterministicAutomaton = field(default_factory=DeterministicAutomaton)
	tagalog_prefixes: Tuple[str, ...] = (
		"mag",
		"nag",
		"pag",
		"ma",
		"na",
		"ka",
		"pa",
		"mang",
		"pang",
		"pan",
		"pinag",
		"ipag",
		"ika",
		"i",
		"um",
	)
	ngram_frequencies: Counter[str] = field(default_factory=Counter)
	top_trigrams: Set[str] = field(default_factory=set)
	english_length_index: Dict[Tuple[str, int], List[str]] = field(default_factory=lambda: defaultdict(list))
	tagalog_length_index: Dict[Tuple[str, int], List[str]] = field(default_factory=lambda: defaultdict(list))
	stemmed_english_words: Set[str] = field(default_factory=set)
	stemmed_tagalog_words: Set[str] = field(default_factory=set)

	def build(self) -> None:
		self.english_automaton.build(self.english_word_list)
		self.tagalog_automaton.build(self.tagalog_word_list)
		self.stemmed_english_words = {simple_stem(word) for word in self.english_word_list}
		self.stemmed_tagalog_words = {simple_stem(word) for word in self.tagalog_word_list}
		self.stemmed_english_automaton.build(self.stemmed_english_words)
		self.stemmed_tagalog_automaton.build(self.stemmed_tagalog_words)
		self.ngram_frequencies = Counter()
		for word in self.stemmed_tagalog_words:
			padded = f"^{word}$"
			for trigram in character_trigrams(padded):
				self.ngram_frequencies[trigram] += 1

		ordered_trigrams = self.ngram_frequencies.most_common()
		if ordered_trigrams:
			keep_count = max(1, int(len(ordered_trigrams) * self.ngram_threshold))
			self.top_trigrams = {trigram for trigram, _ in ordered_trigrams[:keep_count]}
		else:
			self.top_trigrams = set()

		self.english_length_index = defaultdict(list)
		self.tagalog_length_index = defaultdict(list)
		for word in self.english_word_list:
			self.english_length_index[(word[:1], len(word))].append(word)
		for word in self.tagalog_word_list:
			self.tagalog_length_index[(word[:1], len(word))].append(word)

	def retrain(
		self,
		english_word_list: Iterable[str] | None = None,
		tagalog_word_list: Iterable[str] | None = None,
		ngram_threshold: float | None = None,
	) -> None:
		if english_word_list is not None:
			self.english_word_list = {word.lower() for word in english_word_list if word}
		if tagalog_word_list is not None:
			self.tagalog_word_list = {word.lower() for word in tagalog_word_list if word}
		if ngram_threshold is not None:
			self.ngram_threshold = ngram_threshold
		self.build()

	def contains_english(self, word: str) -> bool:
		word = word.lower()
		return self.english_automaton.contains(word) or self.stemmed_english_automaton.contains(simple_stem(word))

	def contains_tagalog(self, word: str) -> bool:
		word = word.lower()
		return self.tagalog_automaton.contains(word) or self.stemmed_tagalog_automaton.contains(simple_stem(word))

	def candidate_pool(self, token: str) -> List[str]:
		language_hint = self.guess_language(token)
		lower_token = token.lower()
		first_letter = lower_token[:1]
		first_two = lower_token[:2]
		length = len(lower_token)
		pool: Set[str] = set()

		# Neutral pool if language is unknown.
		if language_hint == "tagalog":
			primary_words = self.tagalog_word_list
			secondary_words = self.english_word_list
		elif language_hint == "english":
			primary_words = self.english_word_list
			secondary_words = self.tagalog_word_list
		else:
			primary_words = self.tagalog_word_list | self.english_word_list
			secondary_words = set()


		for delta in (-2, -1, 0, 1, 2):
			bucket_length = length + delta
			if bucket_length <= 0:
				continue
			if language_hint == "tagalog":
				pool.update(self.tagalog_length_index.get((first_letter, bucket_length), []))
				pool.update([word for word in self.tagalog_word_list if word.startswith(first_two) and abs(len(word) - length) <= 2])
			else:
				pool.update(self.english_length_index.get((first_letter, bucket_length), []))
				pool.update([word for word in self.english_word_list if word.startswith(first_two) and abs(len(word) - length) <= 2])
		if not pool:
			pool.update(primary_words)
			pool.update([word for word in secondary_words if abs(len(word) - length) <= 1])
		return list(pool)

	def guess_language(self, token: str) -> str:

		"""Return a language hint without over-biasing OOV tokens.

		If the token (or its conservative stem) exists in a lexicon, prefer that
		language. Otherwise return "unknown" so candidate ranking can use a
		neutral pool.
		"""
		lower_token = token.lower()
		if "-" in lower_token:
			parts = [part for part in lower_token.split("-") if part]
			if parts and self.is_tagalog_prefix(parts[0]):
				return "tagalog"

		has_tag = self.contains_tagalog(lower_token) or self.contains_tagalog(simple_stem(lower_token))
		has_eng = self.contains_english(lower_token) or self.contains_english(simple_stem(lower_token))

		if has_tag and not has_eng:
			return "tagalog"
		if has_eng and not has_tag:
			return "english"
		return "unknown"



	def is_tagalog_prefix(self, token: str) -> bool:
		return token.lower() in self.tagalog_prefixes

	def is_valid_hyphenated(self, token: str) -> bool:
		parts = [part.lower() for part in token.split("-") if part]

		if len(parts) != 2:
			return False
		left, right = parts
		if self.is_tagalog_prefix(left) and self.contains_english(right):
			return True
		return self.contains_tagalog(left) and self.contains_tagalog(right)


@dataclass
class DetectedError:
	token: str
	index: int
	reason: str
	previous_token: str = ""
	next_token: str = ""
	candidates: List[str] = field(default_factory=list)


@dataclass
class ErrorDetectionModule:
	modeling: ModelingModule
	ngram_size: int = 3

	def detect(self, tokens: Sequence[str]) -> List[DetectedError]:
		errors: List[DetectedError] = []
		for index, token in enumerate(tokens):
			previous_token = tokens[index - 1] if index > 0 else ""
			next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
			if self._is_valid_token(token, tokens, index):
				continue

			candidates = self._rank_candidates(token)
			errors.append(
				DetectedError(
					token=token,
					index=index,
					reason=self._classify_reason(token),
					previous_token=previous_token,
					next_token=next_token,
					candidates=candidates,
				)
			)
		return errors

	def _is_valid_token(self, token: str, tokens: Sequence[str], index: int) -> bool:
		lower_token = token.lower()
		if self.modeling.contains_english(lower_token) or self.modeling.contains_tagalog(lower_token):
			return True

		if "-" in token and self.modeling.is_valid_hyphenated(token):
			return True

		if index > 0:
			combined = f"{tokens[index - 1].lower()}{lower_token}"
			if self.modeling.is_tagalog_prefix(tokens[index - 1]) and self.modeling.contains_english(lower_token):
				return True
			if self.modeling.contains_tagalog(combined):
				return True

		stemmed = simple_stem(lower_token)
		if self.modeling.stemmed_english_automaton.contains(stemmed) or self.modeling.stemmed_tagalog_automaton.contains(stemmed):
			return True

		if self._has_valid_ngram_context(lower_token):
			return True

		return False

	def _has_valid_ngram_context(self, token: str) -> bool:
		stemmed = simple_stem(token)
		if len(stemmed) < 3:
			return True
		trigrams = character_trigrams(f"^{stemmed}$")
		if not trigrams:
			return False
		for trigram in trigrams:
			if trigram not in self.modeling.top_trigrams:
				return False
		return True

	def _classify_reason(self, token: str) -> str:
		if "-" in token:
			return "hyphenated word mismatch"
		if token != token.lower() and token != token.upper():
			return "case mismatch"
		return "dictionary, code-switching, or n-gram mismatch"

	def _rank_candidates(self, token: str, limit: int = 5) -> List[str]:
		candidates = self.modeling.candidate_pool(token)
		language_hint = self.modeling.guess_language(token)

		preferred_words = self.modeling.tagalog_word_list if language_hint == "tagalog" else self.modeling.english_word_list
		secondary_words = self.modeling.english_word_list if language_hint == "tagalog" else self.modeling.tagalog_word_list
		scored: List[Tuple[int, float, float, str]] = []
		phonetic_token = filipino_phonetic_code(token)

		# Do NOT truncate early based on preferred/secondary membership.
		# That can accidentally discard correct Tagalog candidates once a few
		# non-preferred candidates fill the limit.
		for candidate in candidates:
			candidate_lower = candidate.lower()
			if candidate_lower not in preferred_words and candidate_lower not in secondary_words:
				continue

			distance = levenshtein_distance(token.lower(), candidate_lower)
			if distance > 2:
				continue
			phonetic_similarity = dice_coefficient(phonetic_token, filipino_phonetic_code(candidate_lower))
			sound_similarity = SequenceMatcher(None, token.lower(), candidate_lower).ratio()
			source_penalty = 0 if candidate_lower in preferred_words else 1
			score = (source_penalty, distance, -phonetic_similarity, -sound_similarity, candidate_lower)
			scored.append(score)
		scored.sort()
		selected = [candidate for _, _, _, _, candidate in scored[:limit]]

		if not selected:
			fallback = [word for word in preferred_words if levenshtein_distance(token.lower(), word) <= 2]
			return fallback[:limit]
		return selected


@dataclass
class UserCorrectionModule:
	modeling: ModelingModule

	def suggest(self, token: str, candidates: Sequence[str], previous_token: str = "", next_token: str = "") -> List[str]:
		suggestions: List[str] = []

		merged_suggestions = self._merge_test(token, next_token)
		if merged_suggestions:
			return merged_suggestions[:5]

		split_suggestions = self._split_test(token)
		if split_suggestions:
			return split_suggestions[:5]

		for candidate in self._edit_distance_suggestions(token, candidates):
			if candidate not in suggestions:
				suggestions.append(candidate)

		return suggestions[:5]

	def _merge_test(self, token: str, next_token: str) -> List[str]:
		if not next_token:
			return []
		if not self.modeling.is_tagalog_prefix(token):
			return []
		merged = f"{token}{next_token}"
		if self.modeling.contains_tagalog(merged) or self.modeling.contains_english(merged):
			return [merged]
		return []

	def _split_test(self, token: str) -> List[str]:
		lower_token = token.lower()
		suggestions: List[str] = []
		for index in range(2, len(lower_token) - 1):
			left = lower_token[:index]
			right = lower_token[index:]
			if len(left) < 2 or len(right) < 2:
				continue
			if self.modeling.contains_tagalog(left) and self.modeling.contains_tagalog(right):
				suggestion = f"{left} {right}"
				if suggestion not in suggestions:
					suggestions.append(suggestion)
		return suggestions

	def _edit_distance_suggestions(self, token: str, candidates: Sequence[str]) -> List[str]:
		if not candidates:
			return []
		language_hint = self.modeling.guess_language(token)
		preferred_words = self.modeling.tagalog_word_list if language_hint == "tagalog" else self.modeling.english_word_list
		unique_candidates: List[str] = []
		seen: Set[str] = set()
		for candidate in candidates:
			candidate_lower = candidate.lower()
			if candidate_lower not in preferred_words and candidate_lower not in self.modeling.english_word_list and candidate_lower not in self.modeling.tagalog_word_list:
				continue
			if candidate_lower not in seen:
				seen.add(candidate_lower)
				unique_candidates.append(candidate_lower)

		phonetic_token = filipino_phonetic_code(token)
		scored: List[Tuple[int, float, float, str]] = []
		for candidate in unique_candidates:
			distance = levenshtein_distance(token.lower(), candidate)
			if distance > 2:
				continue
			phonetic_similarity = dice_coefficient(phonetic_token, filipino_phonetic_code(candidate))
			sound_similarity = SequenceMatcher(None, token.lower(), candidate).ratio()
			source_penalty = 0 if candidate in preferred_words else 1
			scored.append((source_penalty, distance, -phonetic_similarity, -sound_similarity, candidate))
		scored.sort()
		return [candidate for _, _, _, _, candidate in scored]


class TextProcessingSystem:
	def __init__(self, english_word_list: Iterable[str], tagalog_word_list: Iterable[str], ngram_size: int = 3) -> None:
		self.modeling = ModelingModule(set(english_word_list), set(tagalog_word_list), ngram_size=ngram_size)
		self.error_detection = ErrorDetectionModule(self.modeling, ngram_size=ngram_size)
		self.user_correction = UserCorrectionModule(self.modeling)

	def build(self) -> None:
		self.modeling.build()

	def process(self, text: str) -> Dict[str, object]:
		tokens = tokenize(text)
		errors = self.error_detection.detect(tokens)

		suggestions = {}
		for error in errors:
			suggestions[error.token] = self.user_correction.suggest(
				error.token,
				error.candidates,
				error.previous_token,
				error.next_token,
			)

		return {
			"input_text": text,
			"normalized_text": normalize_spaces(text),
			"tokens": tokens,
			"words": tokens,
			"errors": errors,
			"suggestions": suggestions,
		}


def build_default_system() -> TextProcessingSystem:
	english_words = load_english_words()
	tagalog_words = load_tagalog_words()

	system = TextProcessingSystem(english_words, tagalog_words)
	system.build()
	return system


def serialize_analysis(result: Dict[str, object]) -> Dict[str, object]:
	errors: List[DetectedError] = result["errors"]  # type: ignore[assignment]
	suggestions: Dict[str, List[str]] = result["suggestions"]  # type: ignore[assignment]

	serialized_errors = []
	for error in errors:
		serialized_errors.append(
			{
				**asdict(error),
				"suggestions": suggestions.get(error.token, []),
			}
		)

	return {
		"input_text": result["input_text"],
		"normalized_text": result["normalized_text"],
		"tokens": result["tokens"],
		"error_count": len(serialized_errors),
		"errors": serialized_errors,
	}


SYSTEM = build_default_system()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
STATIC_DIR = BASE_DIR / "static"


class RequestHandler(BaseHTTPRequestHandler):
	def do_GET(self) -> None:
		if self.path in {"/", "/index.html"}:
			self._send_text(TEMPLATE_PATH.read_text(encoding="utf-8"), content_type="text/html; charset=utf-8")
			return

		if self.path.startswith("/static/"):
			relative_path = self.path.removeprefix("/static/")
			file_path = STATIC_DIR / relative_path
			if file_path.exists() and file_path.is_file():
				content_type = "text/css; charset=utf-8" if file_path.suffix == ".css" else "application/javascript; charset=utf-8"
				self._send_text(file_path.read_text(encoding="utf-8"), content_type=content_type)
				return

		self.send_error(HTTPStatus.NOT_FOUND, "File not found")

	def do_POST(self) -> None:
		if self.path != "/api/check":
			self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
			return

		content_length = int(self.headers.get("Content-Length", "0"))
		payload_raw = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
		try:
			payload = json.loads(payload_raw)
		except json.JSONDecodeError:
			self.send_error(HTTPStatus.BAD_REQUEST, "Invalid JSON")
			return

		text = payload.get("text", "")
		if not isinstance(text, str):
			text = str(text)

		analysis = serialize_analysis(SYSTEM.process(text))
		analysis["word_count"] = len(analysis["tokens"])
		analysis["character_count"] = len(text)
		analysis["ok"] = True
		self._send_json(analysis)

	def _send_text(self, body: str, *, content_type: str) -> None:
		encoded = body.encode("utf-8")
		self.send_response(HTTPStatus.OK)
		self.send_header("Content-Type", content_type)
		self.send_header("Content-Length", str(len(encoded)))
		self.end_headers()
		self.wfile.write(encoded)

	def _send_json(self, payload: Dict[str, object]) -> None:
		encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
		self.send_response(HTTPStatus.OK)
		self.send_header("Content-Type", "application/json; charset=utf-8")
		self.send_header("Content-Length", str(len(encoded)))
		self.end_headers()
		self.wfile.write(encoded)

	def log_message(self, format: str, *args) -> None:  # noqa: A003
		return


def format_report(result: Dict[str, object]) -> str:
	lines = [f"Input: {result['input_text']}", f"Normalized: {result['normalized_text']}", f"Tokens: {', '.join(result['tokens'])}"]

	errors: List[DetectedError] = result["errors"]  # type: ignore[assignment]
	suggestions: Dict[str, List[str]] = result["suggestions"]  # type: ignore[assignment]

	if not errors:
		lines.append("No errors detected.")
		return "\n".join(lines)

	lines.append("Detected errors:")
	for error in errors:
		suggestion_text = ", ".join(suggestions.get(error.token, [])) or "none"
		lines.append(
			f"- token={error.token!r}, index={error.index}, reason={error.reason}, suggestions={suggestion_text}"
		)
	return "\n".join(lines)


def main() -> None:
	server = ThreadingHTTPServer(("127.0.0.1", 5000), RequestHandler)
	print("Serving KOREKARIYAN at http://127.0.0.1:5000")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nShutting down.")
	finally:
		server.server_close()


if __name__ == "__main__":
	main()
