"""Prepare shared candidate partitions without changing original data."""

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "splits_v2"

SEED = 42
SIMILARITY_THRESHOLD = 0.88
REVIEW_FLOOR = 0.88

EXPECTED_FEATURE_HASH = (
    "1c76c1732abf1722ee1646e898f833396"
    "7516ec96161d24951aee45f46d2e41a"
)

csv.field_size_limit(10_000_000)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, fields, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build_similarity_groups(rows):
    """Link sufficiently similar texts and form connected groups."""

    texts = [
        re.sub(r"\s+", " ", row["text"]).strip().casefold()
        for row in rows
    ]

    # Temporary representation for grouping only.
    # These are not additional detector features.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(3, 3),
        binary=True,
        use_idf=False,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float64,
    )
    matrix = vectorizer.fit_transform(texts)

    if np.any(matrix.getnnz(axis=1) == 0):
        raise ValueError("A text has no usable word trigrams.")

    search = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    ).fit(matrix)

    parents = list(range(len(rows)))

    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    pairs = []

    for start in range(0, len(rows), 64):
        distances, neighbours = search.radius_neighbors(
            matrix[start:start + 64],
            radius=1.0 - REVIEW_FLOOR + 1e-12,
            return_distance=True,
            sort_results=True,
        )

        for offset, (row_distances, row_neighbours) in enumerate(
            zip(distances, neighbours, strict=True)
        ):
            left = start + offset

            for distance, right in zip(
                row_distances, row_neighbours, strict=True
            ):
                right = int(right)

                # Exclude self-comparisons and repeated pairs.
                if right <= left:
                    continue

                similarity = 1.0 - float(distance)
                linked = (
                    similarity + 1e-12 >= SIMILARITY_THRESHOLD
                )
                pairs.append((left, right, similarity, linked))

                if linked:
                    left_root = find(left)
                    right_root = find(right)
                    if left_root != right_root:
                        parents[right_root] = left_root

        if start % 1024 == 0 or start + 64 >= len(rows):
            print(
                f"Similarity check: "
                f"{min(start + 64, len(rows))}/{len(rows)}",
                flush=True,
            )

    groups = [find(index) for index in range(len(rows))]
    return groups, pairs


def allocate_partitions(rows, groups):
    """Allocate whole groups while approximately balancing strata."""

    strata = [
        "|".join((
            row["label_origin"],
            row["source"],
            row["category"],
        ))
        for row in rows
    ]

    groups_per_stratum = defaultdict(set)
    for stratum, group in zip(strata, groups, strict=True):
        groups_per_stratum[stratum].add(group)

    if any(len(values) < 5 for values in groups_per_stratum.values()):
        raise ValueError(
            "A stratum has fewer than five independent groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    allocation_folds = [-1] * len(rows)
    placeholder = np.zeros((len(rows), 1), dtype=np.uint8)

    for fold, (_, held_indices) in enumerate(
        splitter.split(placeholder, strata, groups)
    ):
        for index in held_indices:
            index = int(index)
            if allocation_folds[index] != -1:
                raise RuntimeError("A record was allocated twice.")
            allocation_folds[index] = fold

    if -1 in allocation_folds:
        raise RuntimeError("Some records were not allocated.")

    # Three folds for development, one calibration and one test.
    partitions = [
        "test" if fold == 0
        else "calibration" if fold == 1
        else "development"
        for fold in allocation_folds
    ]

    partitions_per_group = defaultdict(set)
    for group, partition in zip(groups, partitions, strict=True):
        partitions_per_group[group].add(partition)

    if any(len(values) != 1 for values in partitions_per_group.values()):
        raise RuntimeError("A similarity group crosses partitions.")

    stratum_counts = Counter(zip(strata, partitions))
    for stratum in set(strata):
        for partition in ("development", "calibration", "test"):
            if stratum_counts[(stratum, partition)] == 0:
                raise RuntimeError(
                    f"Missing stratum in {partition}: {stratum}"
                )

    return allocation_folds, partitions


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Existing split files "
            "will not be overwritten."
        )

    paths = [
        ROOT / "data" / "corpus_features.csv",
        ROOT / "data" / "human_corpus_sampled.csv",
        ROOT / "data" / "llm_corpus_sampled.csv",
    ]
    before = {path.name: digest(path) for path in paths}

    if before["corpus_features.csv"] != EXPECTED_FEATURE_HASH:
        raise ValueError(
            "The feature file differs from the validated version."
        )

    with paths[0].open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "text", "label_origin", "source", "category"}

        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Required metadata columns are missing.")

        rows = sorted(reader, key=lambda row: row["id"])

    if not rows:
        raise ValueError("No records found.")

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Record IDs are not unique.")

    if any(
        row["label_origin"] not in {"human", "llm"}
        or not row["text"].strip()
        for row in rows
    ):
        raise ValueError("Invalid origin or empty text.")

    groups, pairs = build_similarity_groups(rows)
    folds, partitions = allocate_partitions(rows, groups)

    if any(
        linked and partitions[left] != partitions[right]
        for left, right, _, linked in pairs
    ):
        raise RuntimeError("A linked pair crosses partitions.")

    after = {path.name: digest(path) for path in paths}
    if before != after:
        raise RuntimeError("An original dataset changed during execution.")

    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)

    summary = {
        "status": "candidate_partitions_checks_passed",
        "records": len(rows),
        "groups": len(members),
        "multi_record_groups": sum(
            len(indices) > 1 for indices in members.values()
        ),
        "records_in_multi_record_groups": sum(
            len(indices)
            for indices in members.values()
            if len(indices) > 1
        ),
        "largest_group": max(map(len, members.values())),
        "linked_pairs": sum(int(pair[3]) for pair in pairs),
        "group_crossings": 0,
        "linked_pair_crossings": 0,
        "seed": SEED,
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "review_floor": REVIEW_FLOOR,
        "representation": "binary word trigrams, cosine similarity",
        "normalization": "collapse whitespace and casefold; grouping only",
        "grouping": "connected components",
        "allocation": {
            "fold_0": "test",
            "fold_1": "calibration",
            "folds_2_3_4": "development",
        },
        "input_hashes": before,
        "script_sha256": digest(Path(__file__)),
        "partition_counts": dict(Counter(partitions)),
        "source_counts": {
            partition: dict(Counter(
                row["source"]
                for row, role in zip(rows, partitions, strict=True)
                if role == partition
            ))
            for partition in ("development", "calibration", "test")
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)

    write_csv(
        OUTPUT / "manifest.csv",
        [
            "record_id", "group_id", "partition", "allocation_fold",
            "label", "origin", "source", "category",
        ],
        (
            {
                "record_id": row["id"],
                "group_id": f"group_{groups[index]:06d}",
                "partition": partitions[index],
                "allocation_fold": folds[index],
                "label": int(row["label_origin"] == "llm"),
                "origin": row["label_origin"],
                "source": row["source"],
                "category": row["category"],
            }
            for index, row in enumerate(rows)
        ),
    )

    write_csv(
        OUTPUT / "similarity_pairs.csv",
        ["left_id", "right_id", "similarity", "linked"],
        (
            {
                "left_id": rows[left]["id"],
                "right_id": rows[right]["id"],
                "similarity": similarity,
                "linked": int(linked),
            }
            for left, right, similarity, linked in pairs
        ),
    )

    write_csv(
        OUTPUT / "multi_record_groups.csv",
        ["group_id", "records", "origins", "sources"],
        (
            {
                "group_id": f"group_{group:06d}",
                "records": len(indices),
                "origins": "|".join(sorted({
                    rows[index]["label_origin"] for index in indices
                })),
                "sources": "|".join(sorted({
                    rows[index]["source"] for index in indices
                })),
            }
            for group, indices in sorted(members.items())
            if len(indices) > 1
        ),
    )

    # Written last: a failed earlier export has no completion summary.
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved candidate partitions to: {OUTPUT}")
    print("Original dataset hashes unchanged.")
    print("No authorship detector was trained.")


if __name__ == "__main__":
    main()