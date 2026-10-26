from pathlib import Path
import os
import sys
import time

import duckdb
import pyarrow as pa
from unidecode import unidecode


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\durga\OneDrive\Documents\6ab10eb3b23ba_student_resource"
)

TRAIN_DIR = BASE_DIR / "cleaned_data"
TEST_DIR = BASE_DIR / "test_cleaned_data"
OUTPUT_DIR = BASE_DIR / "output"

TRAIN_S1 = TRAIN_DIR / "source1_clean.tsv"
TRAIN_S2 = TRAIN_DIR / "source2_clean.tsv"
TRAIN_S3 = TRAIN_DIR / "source3_clean.tsv"

TEST_S1 = TEST_DIR / "source1_clean.tsv"
TEST_S2 = TEST_DIR / "source2_clean.tsv"
TEST_S3 = TEST_DIR / "source3_clean.tsv"

TRAIN_CANDIDATES = OUTPUT_DIR / "candidate_pairs_v4_1.tsv"
TEST_CANDIDATES = OUTPUT_DIR / "candidate_pairs_test_v4_1.tsv"

TRAIN_OUTPUT = OUTPUT_DIR / "training_features_fast.parquet"
TEST_OUTPUT = OUTPUT_DIR / "testing_features_fast.parquet"

TRAIN_DB = OUTPUT_DIR / "feature_train_fast.duckdb"
TEST_DB = OUTPUT_DIR / "feature_test_fast.duckdb"

TRAIN_TEMP = OUTPUT_DIR / "feature_temp_train_fast"
TEST_TEMP = OUTPUT_DIR / "feature_temp_test_fast"


# ============================================================
# 2. DUCKDB SETTINGS
# ============================================================

MEMORY_LIMIT = "8GB"
THREADS = max(4, min(12, os.cpu_count() or 8))


# ============================================================
# 3. FILE CHECK
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            f"\nFILE NOT FOUND:\n{path}\n"
        )


# ============================================================
# 4. SQL PATH
# ============================================================

def sql_path(path):

    return str(path).replace("\\", "/").replace("'", "''")


# ============================================================
# 5. TRANSLITERATION
# ============================================================

def transliterate_arrow(values):

    values = values.to_pylist()

    result = []

    for value in values:

        if value is None:
            result.append("")

        else:
            result.append(
                unidecode(str(value))
            )

    return pa.array(
        result,
        type=pa.string()
    )


# ============================================================
# 6. DUCKDB CONNECTION
# ============================================================

def create_connection(
    db_path,
    temp_dir
):

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    con = duckdb.connect(
        str(db_path)
    )

    con.execute(
        f"SET memory_limit='{MEMORY_LIMIT}'"
    )

    con.execute(
        f"SET threads={THREADS}"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

    con.execute(
        "SET enable_progress_bar=true"
    )

    con.execute(
        f"SET temp_directory='{sql_path(temp_dir)}'"
    )

    con.create_function(
        "py_translit",
        transliterate_arrow,
        ["VARCHAR"],
        "VARCHAR",
        type="arrow"
    )

    return con


# ============================================================
# 7. FIND GROUND TRUTH
# ============================================================

def find_ground_truth():

    print("\nSearching for ground truth...")

    for path in BASE_DIR.rglob("*.tsv"):

        filename = path.name.lower()

        if filename in {
            "source1_clean.tsv",
            "source2_clean.tsv",
            "source3_clean.tsv",
            "candidate_pairs_v4_1.tsv",
            "candidate_pairs_test_v4_1.tsv"
        }:
            continue

        con = None

        try:

            con = duckdb.connect()

            result = con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_csv_auto(
                    '{sql_path(path)}',
                    delim='\\t',
                    header=true,
                    sample_size=1000
                )
                """
            ).fetchall()

            columns = {
                row[0]
                for row in result
            }

            if (
                "source1_entity_id" in columns
                and
                "matched_entity_ids" in columns
            ):

                print(
                    f"Ground truth found:\n{path}"
                )

                return path

        except Exception:

            pass

        finally:

            if con is not None:
                con.close()

    raise FileNotFoundError(
        "\nGROUND TRUTH NOT FOUND.\n"
    )


# ============================================================
# 8. CREATE SOURCE TABLE
# ============================================================

def create_source_table(
    con,
    table_name,
    source_path
):

    print(
        f"\nLoading {table_name}: "
        f"{source_path.name}"
    )

    path = sql_path(source_path)

    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table_name} AS

        SELECT

            CAST(
                entity_id AS VARCHAR
            ) AS id,

            lower(
                trim(
                    coalesce(
                        business_name_clean,
                        ''
                    )
                )
            ) AS name,

            lower(
                trim(
                    coalesce(
                        business_address_clean,
                        ''
                    )
                )
            ) AS address,

            lower(
                trim(
                    coalesce(
                        country,
                        ''
                    )
                )
            ) AS country,

            regexp_replace(
                lower(
                    trim(
                        coalesce(
                            business_name_clean,
                            ''
                        )
                    )
                ),
                '[^a-z0-9]+',
                '',
                'g'
            ) AS name_compact,

            regexp_replace(
                lower(
                    trim(
                        coalesce(
                            business_address_clean,
                            ''
                        )
                    )
                ),
                '[^a-z0-9]+',
                '',
                'g'
            ) AS address_compact,

            py_translit(
                lower(
                    trim(
                        coalesce(
                            business_name_clean,
                            ''
                        )
                    )
                )
            ) AS translit_name,

            py_translit(
                lower(
                    trim(
                        coalesce(
                            business_address_clean,
                            ''
                        )
                    )
                )
            ) AS translit_address

        FROM read_csv_auto(
            '{path}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"',
            ignore_errors=false
        )
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM {table_name}
        """
    ).fetchone()[0]

    print(
        f"{table_name}: {count:,} rows"
    )


# ============================================================
# 9. LOAD SOURCES
# ============================================================

def load_all_sources(
    con,
    s1_path,
    s2_path,
    s3_path
):

    create_source_table(
        con,
        "s1",
        s1_path
    )

    create_source_table(
        con,
        "s2",
        s2_path
    )

    create_source_table(
        con,
        "s3",
        s3_path
    )

    con.execute(
        """
        CREATE OR REPLACE TABLE targets AS

        SELECT
            id,
            name,
            address,
            country,
            name_compact,
            address_compact,
            translit_name,
            translit_address

        FROM s2

        UNION ALL

        SELECT
            id,
            name,
            address,
            country,
            name_compact,
            address_compact,
            translit_name,
            translit_address

        FROM s3
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM targets
        """
    ).fetchone()[0]

    print(
        f"Combined targets: {count:,} rows"
    )


# ============================================================
# 10. GROUND TRUTH
# ============================================================

def create_ground_truth(
    con,
    ground_truth_path
):

    print("\nLoading ground truth...")

    path = sql_path(
        ground_truth_path
    )

    con.execute(
        f"""
        CREATE OR REPLACE TABLE ground_truth AS

        SELECT DISTINCT

            CAST(
                source1_entity_id
                AS VARCHAR
            ) AS s1_id,

            trim(target_id)
            AS target_id

        FROM
        (
            SELECT

                CAST(
                    source1_entity_id
                    AS VARCHAR
                ) AS source1_entity_id,

                unnest(
                    string_split(
                        coalesce(
                            matched_entity_ids,
                            ''
                        ),
                        ','
                    )
                ) AS target_id

            FROM read_csv_auto(
                '{path}',
                delim='\\t',
                header=true,
                quote='"',
                escape='"',
                ignore_errors=false
            )

            WHERE
                source1_entity_id IS NOT NULL
        )

        WHERE
            trim(target_id) <> ''
        """
    )

    count = con.execute(
        """
        SELECT COUNT(*)
        FROM ground_truth
        """
    ).fetchone()[0]

    print(
        f"Positive pairs: {count:,}"
    )


# ============================================================
# 11. FAST FEATURE QUERY
# ============================================================

def build_feature_query(
    candidate_path,
    training
):

    candidate = sql_path(
        candidate_path
    )

    if training:

        label_expression = """
            CASE
                WHEN gt.s1_id IS NULL
                THEN 0
                ELSE 1
            END AS label,
        """

        gt_join = """
        LEFT JOIN ground_truth gt

            ON c.s1_id = gt.s1_id

            AND c.target_id = gt.target_id
        """

    else:

        label_expression = ""

        gt_join = ""

    return f"""

    WITH candidates AS
    (
        SELECT

            CAST(
                source1_entity_id
                AS VARCHAR
            ) AS s1_id,

            CAST(
                matched_entity_id
                AS VARCHAR
            ) AS target_id

        FROM read_csv_auto(
            '{candidate}',
            delim='\\t',
            header=true,
            quote='"',
            escape='"',
            ignore_errors=false
        )

        WHERE
            source1_entity_id IS NOT NULL

            AND matched_entity_id IS NOT NULL
    ),

    pairs AS
    (
        SELECT

            c.s1_id,

            c.target_id,

            {label_expression}

            s1.name
                AS s1_name,

            t.name
                AS target_name,

            s1.address
                AS s1_address,

            t.address
                AS target_address,

            s1.country
                AS s1_country,

            t.country
                AS target_country,

            s1.name_compact
                AS s1_name_compact,

            t.name_compact
                AS target_name_compact,

            s1.address_compact
                AS s1_address_compact,

            t.address_compact
                AS target_address_compact,

            s1.translit_name
                AS s1_translit_name,

            t.translit_name
                AS target_translit_name,

            s1.translit_address
                AS s1_translit_address,

            t.translit_address
                AS target_translit_address

        FROM candidates c

        INNER JOIN s1
            ON c.s1_id = s1.id

        INNER JOIN targets t
            ON c.target_id = t.id

        {gt_join}
    ),

    cheap AS
    (
        SELECT

            *,

            CASE
                WHEN
                    s1_name != ''
                    AND target_name != ''
                    AND s1_name = target_name
                THEN 1
                ELSE 0
            END
                AS name_exact,

            CASE
                WHEN
                    s1_name_compact != ''
                    AND target_name_compact != ''
                    AND
                    s1_name_compact =
                    target_name_compact
                THEN 1
                ELSE 0
            END
                AS name_compact_exact,

            CASE
                WHEN
                    s1_translit_name != ''
                    AND target_translit_name != ''
                    AND
                    s1_translit_name =
                    target_translit_name
                THEN 1
                ELSE 0
            END
                AS name_translit_exact,

            CASE
                WHEN
                    s1_name != ''
                    AND target_name != ''
                    AND
                    (
                        contains(
                            s1_name,
                            target_name
                        )

                        OR

                        contains(
                            target_name,
                            s1_name
                        )
                    )
                THEN 1
                ELSE 0
            END
                AS name_containment,

            CASE
                WHEN
                    length(s1_name) >= 4
                    AND
                    length(target_name) >= 4
                    AND
                    left(s1_name, 4)
                    =
                    left(target_name, 4)
                THEN 1
                ELSE 0
            END
                AS name_prefix4,

            CASE
                WHEN
                    length(s1_name) >= 6
                    AND
                    length(target_name) >= 6
                    AND
                    left(s1_name, 6)
                    =
                    left(target_name, 6)
                THEN 1
                ELSE 0
            END
                AS name_prefix6,

            CASE
                WHEN
                    length(s1_name) >= 4
                    AND
                    length(target_name) >= 4
                    AND
                    right(s1_name, 4)
                    =
                    right(target_name, 4)
                THEN 1
                ELSE 0
            END
                AS name_suffix4,

            CASE
                WHEN
                    regexp_matches(
                        s1_name,
                        '[^\\x00-\\x7F]'
                    )
                    !=
                    regexp_matches(
                        target_name,
                        '[^\\x00-\\x7F]'
                    )
                THEN 1
                ELSE 0
            END
                AS name_script_mismatch,

            CASE
                WHEN
                    s1_address != ''
                    AND target_address != ''
                    AND
                    s1_address =
                    target_address
                THEN 1
                ELSE 0
            END
                AS address_exact,

            CASE
                WHEN
                    s1_address_compact != ''
                    AND target_address_compact != ''
                    AND
                    s1_address_compact =
                    target_address_compact
                THEN 1
                ELSE 0
            END
                AS address_compact_exact,

            CASE
                WHEN
                    s1_translit_address != ''
                    AND target_translit_address != ''
                    AND
                    s1_translit_address =
                    target_translit_address
                THEN 1
                ELSE 0
            END
                AS address_translit_exact,

            CASE
                WHEN
                    s1_address != ''
                    AND target_address != ''
                    AND
                    (
                        contains(
                            s1_address,
                            target_address
                        )

                        OR

                        contains(
                            target_address,
                            s1_address
                        )
                    )
                THEN 1
                ELSE 0
            END
                AS address_containment,

            CASE
                WHEN
                    length(s1_address) >= 5
                    AND
                    length(target_address) >= 5
                    AND
                    left(s1_address, 5)
                    =
                    left(target_address, 5)
                THEN 1
                ELSE 0
            END
                AS address_prefix5,

            CASE
                WHEN
                    length(s1_address) >= 5
                    AND
                    length(target_address) >= 5
                    AND
                    right(s1_address, 5)
                    =
                    right(target_address, 5)
                THEN 1
                ELSE 0
            END
                AS address_suffix5,

            CASE
                WHEN
                    regexp_extract(
                        s1_address,
                        '^[^0-9]*([0-9]+)',
                        1
                    ) != ''

                    AND

                    regexp_extract(
                        target_address,
                        '^[^0-9]*([0-9]+)',
                        1
                    ) != ''

                    AND

                    regexp_extract(
                        s1_address,
                        '^[^0-9]*([0-9]+)',
                        1
                    )

                    =

                    regexp_extract(
                        target_address,
                        '^[^0-9]*([0-9]+)',
                        1
                    )

                THEN 1
                ELSE 0
            END
                AS house_number_match,

            CASE
                WHEN
                    regexp_extract(
                        s1_address,
                        '([0-9]{{4,8}})',
                        1
                    ) != ''

                    AND

                    regexp_extract(
                        target_address,
                        '([0-9]{{4,8}})',
                        1
                    ) != ''

                    AND

                    regexp_extract(
                        s1_address,
                        '([0-9]{{4,8}})',
                        1
                    )

                    =

                    regexp_extract(
                        target_address,
                        '([0-9]{{4,8}})',
                        1
                    )

                THEN 1
                ELSE 0
            END
                AS postal_code_match,

            CASE
                WHEN
                    s1_country != ''
                    AND target_country != ''
                    AND
                    s1_country =
                    target_country
                THEN 1
                ELSE 0
            END
                AS country_exact,

            length(s1_name)
                AS s1_name_length,

            length(target_name)
                AS target_name_length,

            abs(
                length(s1_name)
                -
                length(target_name)
            )
                AS name_length_difference,

            length(s1_address)
                AS s1_address_length,

            length(target_address)
                AS target_address_length,

            abs(
                length(s1_address)
                -
                length(target_address)
            )
                AS address_length_difference,

            CASE
                WHEN s1_name = ''
                THEN 1
                ELSE 0
            END
                AS s1_name_missing,

            CASE
                WHEN target_name = ''
                THEN 1
                ELSE 0
            END
                AS target_name_missing,

            CASE
                WHEN s1_address = ''
                THEN 1
                ELSE 0
            END
                AS s1_address_missing,

            CASE
                WHEN target_address = ''
                THEN 1
                ELSE 0
            END
                AS target_address_missing,

            CASE
                WHEN s1_country = ''
                THEN 1
                ELSE 0
            END
                AS s1_country_missing,

            CASE
                WHEN target_country = ''
                THEN 1
                ELSE 0
            END
                AS target_country_missing

        FROM pairs
    ),

    scored AS
    (
        SELECT

            *,

            CASE
                WHEN
                    s1_name != ''
                    AND target_name != ''
                    AND
                    (
                        name_exact = 1
                        OR
                        name_compact_exact = 1
                        OR
                        name_translit_exact = 1
                        OR
                        name_containment = 1
                        OR
                        name_script_mismatch = 1
                        OR
                        name_prefix6 = 1
                    )

                THEN
                    jaro_winkler_similarity(
                        s1_name,
                        target_name
                    )

                ELSE 0.0

            END
                AS name_jaro_winkler,

            CASE
                WHEN
                    s1_translit_name != ''
                    AND target_translit_name != ''
                    AND
                    (
                        name_translit_exact = 1
                        OR
                        name_script_mismatch = 1
                    )

                THEN
                    jaro_winkler_similarity(
                        s1_translit_name,
                        target_translit_name
                    )

                ELSE 0.0

            END
                AS name_translit_jaro_winkler,

            CASE
                WHEN
                    s1_address != ''
                    AND target_address != ''
                    AND
                    (
                        address_exact = 1
                        OR
                        address_compact_exact = 1
                        OR
                        address_translit_exact = 1
                        OR
                        address_containment = 1
                        OR
                        house_number_match = 1
                        OR
                        postal_code_match = 1
                    )

                THEN
                    jaro_winkler_similarity(
                        s1_address,
                        target_address
                    )

                ELSE 0.0

            END
                AS address_jaro_winkler,

            CASE
                WHEN
                    s1_translit_address != ''
                    AND target_translit_address != ''
                    AND
                    (
                        address_translit_exact = 1
                        OR
                        name_script_mismatch = 1
                    )

                THEN
                    jaro_winkler_similarity(
                        s1_translit_address,
                        target_translit_address
                    )

                ELSE 0.0

            END
                AS address_translit_jaro_winkler

        FROM cheap
    )

    SELECT

        s1_id,

        target_id,

        {("label," if training else "")}

        name_exact,
        name_compact_exact,
        name_translit_exact,

        name_jaro_winkler,
        name_translit_jaro_winkler,

        name_containment,

        name_prefix4,
        name_prefix6,
        name_suffix4,

        name_script_mismatch,

        address_exact,
        address_compact_exact,
        address_translit_exact,

        address_jaro_winkler,
        address_translit_jaro_winkler,

        address_containment,

        address_prefix5,
        address_suffix5,

        house_number_match,
        postal_code_match,

        country_exact,

        s1_name_length,
        target_name_length,
        name_length_difference,

        s1_address_length,
        target_address_length,
        address_length_difference,

        s1_name_missing,
        target_name_missing,

        s1_address_missing,
        target_address_missing,

        s1_country_missing,
        target_country_missing

    FROM scored

    """


# ============================================================
# 12. WRITE FEATURES
# ============================================================

def write_features(
    con,
    candidate_path,
    output_path,
    training
):

    print("\nGenerating FAST features...")

    print(
        f"Candidates: {candidate_path}"
    )

    print(
        f"Output: {output_path}"
    )

    query = build_feature_query(
        candidate_path,
        training
    )

    start = time.time()

    con.execute(
        f"""
        COPY
        (
            {query}
        )
        TO '{sql_path(output_path)}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE 250000
        )
        """
    )

    elapsed = (
        time.time() - start
    ) / 60

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{sql_path(output_path)}'
        )
        """
    ).fetchone()[0]

    print(
        f"\nFeature rows: {count:,}"
    )

    print(
        f"Feature generation time: "
        f"{elapsed:.2f} minutes"
    )

    print(
        f"Output: {output_path}"
    )

    return count


# ============================================================
# 13. TRAIN
# ============================================================

def run_train():

    print(
        "\n" + "=" * 70
    )

    print(
        "FAST TRAINING FEATURE ENGINEERING"
    )

    print(
        "=" * 70
    )

    start = time.time()

    require_file(TRAIN_S1)
    require_file(TRAIN_S2)
    require_file(TRAIN_S3)
    require_file(TRAIN_CANDIDATES)

    gt_path = find_ground_truth()

    con = create_connection(
        TRAIN_DB,
        TRAIN_TEMP
    )

    try:

        load_all_sources(
            con,
            TRAIN_S1,
            TRAIN_S2,
            TRAIN_S3
        )

        create_ground_truth(
            con,
            gt_path
        )

        count = write_features(
            con,
            TRAIN_CANDIDATES,
            TRAIN_OUTPUT,
            True
        )

        stats = con.execute(
            f"""
            SELECT

                COALESCE(
                    SUM(label),
                    0
                ),

                COUNT(*)
                -
                COALESCE(
                    SUM(label),
                    0
                )

            FROM read_parquet(
                '{sql_path(TRAIN_OUTPUT)}'
            )
            """
        ).fetchone()

        positives = int(
            stats[0]
        )

        negatives = int(
            stats[1]
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "TRAINING COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Total feature rows : "
            f"{count:,}"
        )

        print(
            f"Positive pairs     : "
            f"{positives:,}"
        )

        print(
            f"Negative pairs     : "
            f"{negatives:,}"
        )

        print(
            f"Output             : "
            f"{TRAIN_OUTPUT}"
        )

    finally:

        con.close()

    elapsed = (
        time.time() - start
    ) / 60

    print(
        f"\nTotal time: "
        f"{elapsed:.2f} minutes"
    )


# ============================================================
# 14. TEST
# ============================================================

def run_test():

    print(
        "\n" + "=" * 70
    )

    print(
        "FAST TEST FEATURE ENGINEERING"
    )

    print(
        "=" * 70
    )

    start = time.time()

    require_file(TEST_S1)
    require_file(TEST_S2)
    require_file(TEST_S3)
    require_file(TEST_CANDIDATES)

    con = create_connection(
        TEST_DB,
        TEST_TEMP
    )

    try:

        load_all_sources(
            con,
            TEST_S1,
            TEST_S2,
            TEST_S3
        )

        count = write_features(
            con,
            TEST_CANDIDATES,
            TEST_OUTPUT,
            False
        )

        print(
            "\n" + "=" * 70
        )

        print(
            "TEST FEATURE ENGINEERING COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            f"Total feature rows : "
            f"{count:,}"
        )

        print(
            f"Output             : "
            f"{TEST_OUTPUT}"
        )

    finally:

        con.close()

    elapsed = (
        time.time() - start
    ) / 60

    print(
        f"\nTotal time: "
        f"{elapsed:.2f} minutes"
    )


# ============================================================
# 15. MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:\n"
            "python feature_engineering.py train\n"
            "python feature_engineering.py test\n"
        )

        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "train":

        run_train()

    elif mode == "test":

        run_test()

    else:

        print(
            "ERROR: argument must be "
            "'train' or 'test'"
        )

        sys.exit(1)