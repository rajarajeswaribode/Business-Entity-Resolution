import pandas as pd
import re
import unicodedata
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(r"C:\Users\Hemanth Reddy\Downloads")

TEST_DIR = BASE_DIR / "student_resource" / "dataset" / "test"

OUTPUT_DIR = BASE_DIR / "test_cleaned_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
SOURCE3_PATH = TEST_DIR / "test_source3.tsv"


# ============================================================
# 2. UNICODE-SAFE PUNCTUATION CLEANING
# ============================================================

def remove_punctuation_keep_unicode(text):
    """
    Remove punctuation/symbols while preserving:
    - Unicode letters
    - Unicode combining marks
    - numbers
    - spaces

    This is important for:
    Hindi, Telugu, Kannada, Tamil, etc.
    """

    result = []

    for char in text:

        category = unicodedata.category(char)

        # Keep letters
        if category.startswith("L"):
            result.append(char)

        # Keep combining marks
        elif category.startswith("M"):
            result.append(char)

        # Keep numbers
        elif category.startswith("N"):
            result.append(char)

        # Keep whitespace
        elif category.startswith("Z"):
            result.append(" ")

        # Everything else = punctuation/symbol
        else:
            result.append(" ")

    return "".join(result)


# ============================================================
# 3. NORMALIZE BUSINESS NAME
# ============================================================

def normalize_name(text):

    # Missing value
    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive matching
    text = text.casefold()

    # Convert & to "and"
    text = text.replace("&", " and ")

    # Remove apostrophes without creating an extra word
    # Example:
    # O'Reilly -> oreilly
    # Moyna's -> moynas
    text = text.replace("'", "")
    text = text.replace("’", "")

    # Remove punctuation but KEEP Unicode letters/marks
    text = remove_punctuation_keep_unicode(text)

    # Replace multiple spaces with one
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 4. NORMALIZE ADDRESS
# ============================================================

def normalize_address(text):

    # Missing value
    if pd.isna(text):
        return ""

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Case-insensitive matching
    text = text.casefold()

    # Convert & to "and"
    text = text.replace("&", " and ")

    # IMPORTANT:
    # Keep address components separate.
    #
    # 570/13  -> 570 13
    # 104/1/1 -> 104 1 1
    # 2260-5  -> 2260 5
    #
    text = re.sub(r"[/\-]", " ", text)

    # Remove apostrophes
    text = text.replace("'", "")
    text = text.replace("’", "")

    # Remove punctuation while preserving Unicode
    text = remove_punctuation_keep_unicode(text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 5. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print("\nLoading Source 1...")
source1 = pd.read_csv(
    SOURCE1_PATH,
    sep="\t",
    dtype=str
)

print("Source 1 rows:", len(source1))


print("\nLoading Source 2...")
source2 = pd.read_csv(
    SOURCE2_PATH,
    sep="\t",
    dtype=str
)

print("Source 2 rows:", len(source2))


print("\nLoading Source 3...")
source3 = pd.read_csv(
    SOURCE3_PATH,
    sep="\t",
    dtype=str
)

print("Source 3 rows:", len(source3))


# ============================================================
# 6. DISPLAY ORIGINAL COLUMNS
# ============================================================

print("\n" + "=" * 70)
print("COLUMNS")
print("=" * 70)

print("\nSource 1 columns:")
print(source1.columns.tolist())

print("\nSource 2 columns:")
print(source2.columns.tolist())

print("\nSource 3 columns:")
print(source3.columns.tolist())


# ============================================================
# 7. CLEAN BUSINESS NAMES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING BUSINESS NAMES")
print("=" * 70)

print("\nCleaning Source 1 business names...")
source1["business_name_clean"] = (
    source1["business_name"]
    .apply(normalize_name)
)

print("Cleaning Source 2 business names...")
source2["business_name_clean"] = (
    source2["business_name"]
    .apply(normalize_name)
)

print("Cleaning Source 3 business names...")
source3["business_name_clean"] = (
    source3["business_name"]
    .apply(normalize_name)
)


# ============================================================
# 8. CLEAN ADDRESSES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZING ADDRESSES")
print("=" * 70)

print("\nCleaning Source 1 addresses...")
source1["business_address_clean"] = (
    source1["business_address"]
    .apply(normalize_address)
)

print("Cleaning Source 2 addresses...")
source2["business_address_clean"] = (
    source2["business_address"]
    .apply(normalize_address)
)

print("Cleaning Source 3 addresses...")
source3["business_address_clean"] = (
    source3["business_address"]
    .apply(normalize_address)
)


# ============================================================
# 9. QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("QUALITY CHECK")
print("=" * 70)


print("\nSource 1:")
print(
    "Missing cleaned names:",
    source1["business_name_clean"].eq("").sum()
)

print(
    "Missing cleaned addresses:",
    source1["business_address_clean"].eq("").sum()
)


print("\nSource 2:")
print(
    "Missing cleaned names:",
    source2["business_name_clean"].eq("").sum()
)

print(
    "Missing cleaned addresses:",
    source2["business_address_clean"].eq("").sum()
)


print("\nSource 3:")
print(
    "Missing cleaned names:",
    source3["business_name_clean"].eq("").sum()
)

print(
    "Missing cleaned addresses:",
    source3["business_address_clean"].eq("").sum()
)


# ============================================================
# 10. TEST NORMALIZATION WITH EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZATION TESTS")
print("=" * 70)


# ----------------------------
# Address tests
# ----------------------------

address_tests = [
    "KH NO. -570/13, NEW DELHI",
    "104/1/1 ERANDWANE",
    "2260- Housecreek Trail",
    "85 Wayne Avenue, Ticonderoga, NY",
    "G-3/571, GULMOHAR COLONY",
    "63/2275/7, ALHIND TOWER, FIRST FLOOR",
    "H.NO 204 C ROAD HOSHIARPUR, PUNJAB",
    "##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri",
    "ಬೆಂಗಳೂರು ಕರ್ನಾಟಕ",
    "85 Wanye Avenue, Ticonderoga Townshiip, New York"
]


print("\nADDRESS TESTS:")

for address in address_tests:

    print("\nOriginal :", address)
    print("Cleaned  :", normalize_address(address))


# ----------------------------
# Name tests
# ----------------------------

name_tests = [
    "Orelee's Barbershop",
    "B+ Retail Inc",
    "Sri Lakshmi & Sons",
    "Moyna's Coffee",
    "ABC & Sons Pvt. Ltd.",
    "Prime Money",
    "Maure Williams Colombier Inc",
    "Maure Wilblims Colombier",
    "राम मार्केटिंग प्राइवेट लिमिटेड",
    "आदित्य प्रॉपर्टीज एलएलपी",
    "ಕರ್ನಾಟಕ"
]


print("\n" + "=" * 70)
print("BUSINESS NAME TESTS")
print("=" * 70)

for name in name_tests:

    print("\nOriginal :", name)
    print("Cleaned  :", normalize_name(name))


# ============================================================
# 11. SHOW FIRST 10 CLEANED RECORDS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEANED RECORDS")
print("=" * 70)


print("\nSOURCE 1:")
print(
    source1[
        [
            "entity_id",
            "business_name",
            "business_name_clean",
            "business_address",
            "business_address_clean",
            "country"
        ]
    ].head(10).to_string(index=False)
)


print("\nSOURCE 2:")
print(
    source2[
        [
            "entity_id",
            "business_name",
            "business_name_clean",
            "business_address",
            "business_address_clean",
            "country"
        ]
    ].head(10).to_string(index=False)
)


print("\nSOURCE 3:")
print(
    source3[
        [
            "entity_id",
            "business_name",
            "business_name_clean",
            "business_address",
            "business_address_clean",
            "country"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 12. SAVE CLEANED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING CLEANED DATA")
print("=" * 70)


source1_output = OUTPUT_DIR / "source1_clean.tsv"
source2_output = OUTPUT_DIR / "source2_clean.tsv"
source3_output = OUTPUT_DIR / "source3_clean.tsv"


source1.to_csv(
    source1_output,
    sep="\t",
    index=False
)

source2.to_csv(
    source2_output,
    sep="\t",
    index=False
)

source3.to_csv(
    source3_output,
    sep="\t",
    index=False
)


print("\nSource 1 saved to:")
print(source1_output)

print("\nSource 2 saved to:")
print(source2_output)

print("\nSource 3 saved to:")
print(source3_output)


# ============================================================
# 13. FINAL ROW COUNT CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

print("\nSource 1 rows:", len(source1))
print("Source 2 rows:", len(source2))
print("Source 3 rows:", len(source3))

print("\nOriginal TSV files were NOT modified.")

print("\nCleaning completed successfully!")

print("\nNext step:")
print("Candidate generation / blocking")