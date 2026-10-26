import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

TRAIN_FILE = Path(
    r"C:\Users\dhath\OneDrive\Documents\ML_Challenge\output\training_features_fast.parquet"
)

TEST_FILE = Path(
    r"C:\Users\dhath\OneDrive\Documents\ML_Challenge\output\testing_features_fast.parquet"
)

GROUND_TRUTH_FILE = Path(
    r"C:\Users\dhath\OneDrive\Documents\ML_Challenge\student_resource\dataset\train\train_ground_truth.tsv"
)

MODEL_FILE = Path(
    r"C:\Users\dhath\OneDrive\Documents\ML_Challenge\output\lightgbm_model.txt"
)

OUTPUT_FILE = Path(
    r"C:\Users\dhath\OneDrive\Documents\ML_Challenge\output\matching_results.tsv"
)

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 100_000

MAX_POSITIVE = 1_000_000
MAX_NEGATIVE = 3_000_000

RANDOM_SEED = 42

# Probability threshold for final matching
MATCH_THRESHOLD = 0.50

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "name_exact",
    "name_compact_exact",
    "name_translit_exact",
    "name_jaro_winkler",
    "name_translit_jaro_winkler",
    "name_containment",
    "name_prefix4",
    "name_prefix6",
    "name_suffix4",
    "name_script_mismatch",

    "address_exact",
    "address_compact_exact",
    "address_translit_exact",
    "address_jaro_winkler",
    "address_translit_jaro_winkler",
    "address_containment",
    "address_prefix5",
    "address_suffix5",
    "house_number_match",
    "postal_code_match",
    "country_exact",

    "s1_name_length",
    "target_name_length",
    "name_length_difference",

    "s1_address_length",
    "target_address_length",
    "address_length_difference",

    "s1_name_missing",
    "target_name_missing",
    "s1_address_missing",
    "target_address_missing",
    "s1_country_missing",
    "target_country_missing",
]


# ============================================================
# 1. CREATE TRAINING SAMPLE
# ============================================================

print("=" * 70)
print("CREATING TRAINING SAMPLE")
print("=" * 70)

pf = pq.ParquetFile(TRAIN_FILE)

positive_parts = []
negative_parts = []

positive_count = 0
negative_count = 0

train_columns = ["s1_id", "target_id", "label"] + FEATURES

for batch_number, batch in enumerate(
    pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=train_columns
    )
):

    df = batch.to_pandas()

    # --------------------------------------------------------
    # POSITIVES
    # --------------------------------------------------------

    pos = df[df["label"] == 1]

    remaining_pos = MAX_POSITIVE - positive_count

    if remaining_pos > 0 and len(pos) > 0:

        if len(pos) > remaining_pos:
            indices = rng.choice(
                len(pos),
                size=remaining_pos,
                replace=False
            )
            pos = pos.iloc[indices]

        positive_parts.append(pos)
        positive_count += len(pos)

    # --------------------------------------------------------
    # NEGATIVES
    # --------------------------------------------------------

    neg = df[df["label"] == 0]

    remaining_neg = MAX_NEGATIVE - negative_count

    if remaining_neg > 0 and len(neg) > 0:

        if len(neg) > remaining_neg:
            indices = rng.choice(
                len(neg),
                size=remaining_neg,
                replace=False
            )
            neg = neg.iloc[indices]

        negative_parts.append(neg)
        negative_count += len(neg)

    if batch_number % 100 == 0:
        print(
            f"Batch {batch_number} | "
            f"Positive: {positive_count:,} | "
            f"Negative: {negative_count:,}"
        )

    if (
        positive_count >= MAX_POSITIVE
        and negative_count >= MAX_NEGATIVE
    ):
        break


# ============================================================
# 2. COMBINE TRAINING DATA
# ============================================================

print("\nCombining training data...")

train_df = pd.concat(
    positive_parts + negative_parts,
    ignore_index=True
)

del positive_parts
del negative_parts

train_df = train_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)

print("\nTraining sample shape:", train_df.shape)

print("\nLabel distribution:")
print(train_df["label"].value_counts())


# ============================================================
# 3. X AND Y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df["label"]

del train_df


# ============================================================
# 4. LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LIGHTGBM")
print("=" * 70)

model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=500,
    learning_rate=0.05,

    num_leaves=31,
    max_depth=-1,

    subsample=0.8,
    colsample_bytree=0.8,

    reg_alpha=0.1,
    reg_lambda=0.1,

    random_state=RANDOM_SEED,

    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("\nTraining completed.")


# ============================================================
# 5. SAVE MODEL
# ============================================================

print("\nSaving model...")

MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

model.booster_.save_model(str(MODEL_FILE))

print("Model saved to:")
print(MODEL_FILE)


# ============================================================
# 6. TRAINING METRICS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING METRICS")
print("=" * 70)

train_probabilities = model.predict_proba(X_train)[:, 1]

train_predictions = (
    train_probabilities >= MATCH_THRESHOLD
).astype(int)

y_true = y_train.to_numpy()

TP = np.sum((y_true == 1) & (train_predictions == 1))
TN = np.sum((y_true == 0) & (train_predictions == 0))
FP = np.sum((y_true == 0) & (train_predictions == 1))
FN = np.sum((y_true == 1) & (train_predictions == 0))

precision = (
    TP / (TP + FP)
    if (TP + FP) > 0
    else 0
)

recall = (
    TP / (TP + FN)
    if (TP + FN) > 0
    else 0
)

f05 = (
    (1.25 * precision * recall)
    / (0.25 * precision + recall)
    if (0.25 * precision + recall) > 0
    else 0
)

accuracy = (
    (TP + TN) / (TP + TN + FP + FN)
)

print(f"\nThreshold       : {MATCH_THRESHOLD}")
print(f"TP              : {TP:,}")
print(f"TN              : {TN:,}")
print(f"FP              : {FP:,}")
print(f"FN              : {FN:,}")

print(f"\nPrecision       : {precision:.6f}")
print(f"Recall          : {recall:.6f}")
print(f"F0.5            : {f05:.6f}")
print(f"Accuracy        : {accuracy:.6f}")


# ============================================================
# 7. PREDICT TEST DATA
# ============================================================

print("\n" + "=" * 70)
print("PREDICTING TEST DATA")
print("=" * 70)

test_pf = pq.ParquetFile(TEST_FILE)

# Store ALL candidates above threshold
# for every Source 1 entity.
matches = {}

# Blocking statistics
total_candidate_pairs = 0
candidate_counts = {}

test_columns = ["s1_id", "target_id"] + FEATURES

for batch_number, batch in enumerate(
    test_pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=test_columns
    )
):

    test_df = batch.to_pandas()

    total_candidate_pairs += len(test_df)

    X_test = test_df[FEATURES]

    probabilities = model.predict_proba(X_test)[:, 1]

    test_df["probability"] = probabilities

    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

    batch_counts = test_df["s1_id"].value_counts()

    for s1_id, count in batch_counts.items():

        candidate_counts[s1_id] = (
            candidate_counts.get(s1_id, 0) + count
        )

    # --------------------------------------------------------
    # Keep candidates above threshold
    # --------------------------------------------------------

    matched = test_df[
        test_df["probability"] >= MATCH_THRESHOLD
    ]

    for row in matched.itertuples(index=False):

        s1_id = row.s1_id
        target_id = row.target_id

        if s1_id not in matches:
            matches[s1_id] = []

        matches[s1_id].append(target_id)

    if batch_number % 100 == 0:

        print(
            f"Processed test batches: {batch_number:,} | "
            f"Candidate pairs: {total_candidate_pairs:,} | "
            f"S1 entities: {len(candidate_counts):,}"
        )


# ============================================================
# 8. BLOCKING METRICS
# ============================================================

print("\n" + "=" * 70)
print("BLOCKING METRICS")
print("=" * 70)

# Number of S1 entities in candidate file
num_s1_entities = len(candidate_counts)

print(
    f"\nNumber of Source 1 entities : "
    f"{num_s1_entities:,}"
)

print(
    f"Candidate pairs              : "
    f"{total_candidate_pairs:,}"
)

if num_s1_entities > 0:

    candidate_values = np.array(
        list(candidate_counts.values())
    )

    average_candidates = np.mean(candidate_values)
    median_candidates = np.median(candidate_values)
    maximum_candidates = np.max(candidate_values)

    print(
        f"Average candidates/S1       : "
        f"{average_candidates:.2f}"
    )

    print(
        f"Median candidates/S1        : "
        f"{median_candidates:.2f}"
    )

    print(
        f"Maximum candidates/S1       : "
        f"{maximum_candidates:,}"
    )


# ============================================================
# 9. BLOCKING RECALL
# ============================================================

print("\nCalculating blocking recall...")

# Count all true matches that survived blocking.
# label=1 rows in training_features are the true matches
# that were present in the generated candidate set.

training_pf = pq.ParquetFile(TRAIN_FILE)

blocked_true_matches = 0

for batch in training_pf.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["label"]
):

    labels = batch.column("label").to_numpy()

    blocked_true_matches += np.sum(labels == 1)


# Count total true matches from ground truth
ground_truth = pd.read_csv(
    GROUND_TRUTH_FILE,
    sep="\t",
    dtype=str
)

total_true_matches = 0

for value in ground_truth["matched_entity_ids"]:

    if pd.isna(value) or str(value).strip() == "":
        continue

    ids = str(value).split(",")

    total_true_matches += len(
        [x for x in ids if x.strip()]
    )


blocking_recall = (
    blocked_true_matches / total_true_matches
    if total_true_matches > 0
    else 0
)

print(
    f"\nTrue matches in ground truth : "
    f"{total_true_matches:,}"
)

print(
    f"True matches retained         : "
    f"{blocked_true_matches:,}"
)

print(
    f"Blocking Recall               : "
    f"{blocking_recall:.6f}"
)


# ============================================================
# 10. REDUCTION RATIO
# ============================================================

# We need the number of Source 2 and Source 3 records.
# These can be obtained from the candidate file only if every
# possible pair count is known. We therefore read the training
# source files.

TRAIN_SOURCE1 = GROUND_TRUTH_FILE.parent / "train_source1.tsv"
TRAIN_SOURCE2 = GROUND_TRUTH_FILE.parent / "train_source2.tsv"
TRAIN_SOURCE3 = GROUND_TRUTH_FILE.parent / "train_source3.tsv"

try:

    n_source1 = sum(
        1
        for _ in open(TRAIN_SOURCE1, "r", encoding="utf-8")
    ) - 1

    n_source2 = sum(
        1
        for _ in open(TRAIN_SOURCE2, "r", encoding="utf-8")
    ) - 1

    n_source3 = sum(
        1
        for _ in open(TRAIN_SOURCE3, "r", encoding="utf-8")
    ) - 1

    total_possible_pairs = (
        n_source1 * (n_source2 + n_source3)
    )

    reduction_ratio = (
        1 -
        total_candidate_pairs / total_possible_pairs
    )

    print("\n" + "=" * 70)
    print("REDUCTION RATIO")
    print("=" * 70)

    print(
        f"\nSource 1 records       : {n_source1:,}"
    )

    print(
        f"Source 2 records       : {n_source2:,}"
    )

    print(
        f"Source 3 records       : {n_source3:,}"
    )

    print(
        f"Possible pairs         : "
        f"{total_possible_pairs:,}"
    )

    print(
        f"Candidate pairs        : "
        f"{total_candidate_pairs:,}"
    )

    print(
        f"Reduction Ratio        : "
        f"{reduction_ratio:.6%}"
    )

except FileNotFoundError:

    print(
        "\nTraining source files not found."
    )

    print(
        "Reduction ratio could not be calculated."
    )


# ============================================================
# 11. FINAL MATCH STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MATCH STATISTICS")
print("=" * 70)

# Number of S1 entities with at least one prediction
predicted_non_singletons = len(matches)

# Number of S1 entities with no prediction
predicted_singletons = (
    num_s1_entities - predicted_non_singletons
)

total_predicted_matches = sum(
    len(v) for v in matches.values()
)

print(
    f"\nS1 entities                    : "
    f"{num_s1_entities:,}"
)

print(
    f"S1 entities with matches       : "
    f"{predicted_non_singletons:,}"
)

print(
    f"Predicted singletons           : "
    f"{predicted_singletons:,}"
)

print(
    f"Total predicted matches        : "
    f"{total_predicted_matches:,}"
)

if predicted_non_singletons > 0:

    print(
        f"Average matches per matched S1: "
        f"{total_predicted_matches / predicted_non_singletons:.2f}"
    )


# ============================================================
# 12. CREATE FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CREATING FINAL RESULTS")
print("=" * 70)

results = []

# IMPORTANT:
# Every Source 1 entity must appear exactly once.
# Therefore iterate over candidate_counts, not matches.

for s1_id in candidate_counts:

    target_ids = matches.get(s1_id, [])

    # Remove duplicates
    target_ids = list(dict.fromkeys(target_ids))

    results.append({
        "source1_entity_id": s1_id,
        "matched_entity_ids": ",".join(target_ids)
    })


results_df = pd.DataFrame(
    results,
    columns=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)


# ============================================================
# 13. SAVE TSV
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print("\nFinished.")

print(
    "Number of predictions:",
    len(results_df)
)

print("\nFirst 10 predictions:")

print(
    results_df.head(10).to_string(
        index=False
    )
)

print("\nOutput file:")
print(OUTPUT_FILE)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL METRICS SUMMARY")
print("=" * 70)

print(f"""
Threshold              : {MATCH_THRESHOLD}

TP                     : {TP:,}
TN                     : {TN:,}
FP                     : {FP:,}
FN                     : {FN:,}

Precision              : {precision:.6f}
Recall                 : {recall:.6f}
F0.5                   : {f05:.6f}

Blocking Recall        : {blocking_recall:.6f}

Candidate Pairs        : {total_candidate_pairs:,}
Average Candidates/S1  : {average_candidates:.2f}
Median Candidates/S1   : {median_candidates:.2f}
Maximum Candidates/S1  : {maximum_candidates:,}

Predicted Matches      : {total_predicted_matches:,}
Predicted Singletons   : {predicted_singletons:,}

Output                 : {OUTPUT_FILE}
""")