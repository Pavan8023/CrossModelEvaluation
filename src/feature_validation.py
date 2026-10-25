"""Read-only sample validation. Does not modify dataset files."""

import ast
import csv
import hashlib
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import spacy

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT.parent
    / "cross model paper"
    / "cross-model-phishing"
    / "code"
    / "05_extract_features.py"
)

FEATURES = [
    "ttr", "mean_word_len", "mean_sentence_len_tokens", "yules_k",
    "clause_density", "noun_ratio", "verb_ratio", "mean_parse_depth",
    "imperative_count", "first_person_ratio", "second_person_ratio",
    "politeness_density", "urgency_density", "url_density",
    "cta_density", "authority_density", "time_pressure_density",
]

FOUR_DECIMALS = {
    "ttr", "noun_ratio", "verb_ratio",
    "first_person_ratio", "second_person_ratio",
}

LEXICON_NAMES = {
    "politeness_density": "POLITENESS_TERMS",
    "urgency_density": "URGENCY_TERMS",
    "cta_density": "CTA_TERMS",
    "authority_density": "AUTHORITY_TERMS",
    "time_pressure_density": "TIME_PRESSURE_TERMS",
}

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
SECOND_PERSON = {"you", "your", "yours", "yourself", "yourselves"}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")

csv.field_size_limit(10_000_000)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_reference_lexicons():
    """Read literal dictionaries without executing reference code."""
    tree = ast.parse(REFERENCE.read_text(encoding="utf-8-sig"))
    constants = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in LEXICON_NAMES.values()
                ):
                    constants[target.id] = ast.literal_eval(node.value)

    missing = set(LEXICON_NAMES.values()) - constants.keys()
    if missing:
        raise ValueError(f"Missing reference dictionaries: {missing}")

    return {
        feature: constants[name]
        for feature, name in LEXICON_NAMES.items()
    }


def count_terms(text_lower, terms):
    """Reproduce the inspected reference matching convention."""
    return sum(
        text_lower.count(term)
        if " " in term
        else len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower))
        for term in terms
    )


def recompute(text, doc, lexicons):
    tokens = [token for token in doc if not token.is_space]
    words = [token.text.lower() for token in tokens if token.is_alpha]
    sentences = list(doc.sents)

    if not words or not tokens or not sentences:
        raise ValueError("Cannot validate an empty linguistic representation.")

    n_words = len(words)
    n_tokens = len(tokens)
    n_sentences = len(sentences)
    frequencies = Counter(words)

    first_tokens = [
        next((token for token in sent if not token.is_space), None)
        for sent in sentences
    ]

    values = {
        "ttr": len(frequencies) / n_words,
        "mean_word_len": sum(map(len, words)) / n_words,
        "mean_sentence_len_tokens": n_tokens / n_sentences,
        "yules_k": (
            10000
            * (sum(count ** 2 for count in frequencies.values()) - n_words)
            / n_words ** 2
        ),
        "clause_density": sum(
            token.dep_ in {"ROOT", "ccomp", "advcl", "relcl", "xcomp"}
            for token in doc
        ) / n_sentences,
        "noun_ratio": sum(t.pos_ == "NOUN" for t in tokens) / n_tokens,
        "verb_ratio": sum(t.pos_ == "VERB" for t in tokens) / n_tokens,
        # Includes whitespace tokens to match the reference convention.
        # Ancestor counts independently express distance to the root.
        "mean_parse_depth": sum(
            min(200, len(list(token.ancestors))) for token in doc
        ) / len(doc),
        "imperative_count": sum(
            token is not None
            and token.tag_ == "VB"
            and token.dep_ in {"ROOT", "ccomp"}
            for token in first_tokens
        ),
        "first_person_ratio": sum(
            word in FIRST_PERSON for word in words
        ) / n_words,
        "second_person_ratio": sum(
            word in SECOND_PERSON for word in words
        ) / n_words,
        "url_density": 100 * len(URL_PATTERN.findall(text)) / n_words,
    }

    for feature, terms in lexicons.items():
        values[feature] = 100 * count_terms(text.lower(), terms) / n_words

    for feature in FEATURES:
        decimals = (
            4 if feature in FOUR_DECIMALS
            else 0 if feature == "imperative_count"
            else 3
        )
        values[feature] = round(values[feature], decimals)

    return values


def main():
    paths = [
        ROOT / "data" / "corpus_features.csv",
        ROOT / "data" / "human_corpus_sampled.csv",
        ROOT / "data" / "llm_corpus_sampled.csv",
    ]
    before = {str(path): sha256(path) for path in paths}

    rows = read_csv(paths[0])
    raw_rows = read_csv(paths[1]) + read_csv(paths[2])

    if not rows:
        raise ValueError("Feature table is empty.")

    missing = set(FEATURES) - rows[0].keys()
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Feature IDs are not unique.")

    raw_by_text = {row["text"]: row for row in raw_rows}
    if len(raw_by_text) != len(raw_rows):
        raise ValueError("Raw texts are not unique; join needs review.")

    if (
        len(rows) != len(raw_rows)
        or len({row["text"] for row in rows}) != len(rows)
        or {row["text"] for row in rows} != set(raw_by_text)
    ):
        raise ValueError("Feature-to-raw text alignment failed.")

    for row in rows:
        for feature in FEATURES:
            value = float(row[feature])
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid value: {row['id']}, {feature}, {value}"
                )

    # Five records per source/category group, plus each feature's extremes.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source"], row["category"])].append(row)

    rng = random.Random(42)
    selected_ids = set()

    for key in sorted(groups):
        bucket = sorted(groups[key], key=lambda row: row["id"])
        selected_ids.update(
            row["id"] for row in rng.sample(bucket, min(5, len(bucket)))
        )

    for feature in FEATURES:
        selected_ids.add(
            min(rows, key=lambda row: float(row[feature]))["id"]
        )
        selected_ids.add(
            max(rows, key=lambda row: float(row[feature]))["id"]
        )

    sample = sorted(
        (row for row in rows if row["id"] in selected_ids),
        key=lambda row: row["id"],
    )

    lexicons = load_reference_lexicons()
    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    tasks = []
    for row in sample:
        variants = {
            "stored_text": row["text"],
            "collapsed_whitespace": re.sub(r"\s+", " ", row["text"]).strip(),
            "body_only": raw_by_text[row["text"]]["body"],
        }
        tasks.extend(
            (row, variant, text) for variant, text in variants.items()
        )

    matches = defaultdict(Counter)
    examples = defaultdict(list)

    documents = nlp.pipe(
        (text for _, _, text in tasks),
        batch_size=32,
    )

    for (row, variant, text), doc in zip(tasks, documents):
        calculated = recompute(text, doc, lexicons)

        for feature in FEATURES:
            stored = float(row[feature])
            equal = math.isclose(
                stored, calculated[feature],
                rel_tol=0,
                abs_tol=1e-9,
            )
            matches[variant][feature] += int(equal)

            if (
                variant == "stored_text"
                and not equal
                and len(examples[feature]) < 3
            ):
                examples[feature].append({
                    "id": row["id"],
                    "stored": stored,
                    "recomputed": calculated[feature],
                })

    after = {str(path): sha256(path) for path in paths}
    if before != after:
        raise RuntimeError("An input file changed during validation.")

    print("Python/spaCy model provenance:")
    import sys
    print("Python:", sys.version)
    print("spaCy:", spacy.__version__)
    print("Pipeline:", nlp.meta.get("name"), nlp.meta.get("version"))
    print("Reference SHA256:", sha256(REFERENCE))
    print("Input SHA256:", before)
    print("Sample size:", len(sample))
    print("Sample IDs:", [row["id"] for row in sample])

    print("\nMATCH COUNTS — each denominator is the sample size")
    print("feature | stored text | collapsed whitespace | body only")
    for feature in FEATURES:
        counts = [
            matches[variant][feature]
            for variant in (
                "stored_text", "collapsed_whitespace", "body_only"
            )
        ]
        print(feature, *counts, sep=" | ")

    print("\nStored-text mismatch examples:")
    for feature, records in examples.items():
        print(feature, records)

    print("\nOriginal inputs unchanged.")
    print("Sample agreement does not establish full-corpus validity.")


if __name__ == "__main__":
    main()