"""
Goodreads Toy Dataset Generator
================================
Generates 10,000 simulated user-book interactions in Line-delimited JSON format,
mimicking the structure of the real Goodreads dataset.

Author  : Senior Data Engineer
Purpose : Recommendation System prototyping & graph-based modeling
"""

import json
import random
import uuid
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
TOTAL_INTERACTIONS = 10_000
NUM_USERS          = 500     # unique user pool
NUM_BOOKS          = 200     # unique book pool
OUTPUT_FILE        = "goodreads_toy_10k.json"

# Seed for reproducibility
random.seed(42)

# ─────────────────────────────────────────────
# 2. ENTITY POOLS
# ─────────────────────────────────────────────

def make_hex_id() -> str:
    """Generate a random 32-character hex string (like a MD5 hash)."""
    return uuid.uuid4().hex  # 32-char hex, no hyphens

# Pre-generate fixed pools so the same IDs recur across interactions
USER_POOL = [make_hex_id() for _ in range(NUM_USERS)]
BOOK_POOL = [str(random.randint(1_000_000, 9_999_999)) for _ in range(NUM_BOOKS)]

# ─────────────────────────────────────────────
# 3. DATE HELPERS
# ─────────────────────────────────────────────

# Goodreads date format example: "Sat Jul 09 07:32:33 -0700 2016"
GOODREADS_FMT = "%a %b %d %H:%M:%S %z %Y"

# Fixed timezone offsets used in Goodreads exports
TZ_OFFSETS = [
    timezone(timedelta(hours=h))
    for h in [-8, -7, -5, -4, 0, 1, 2, 5, 8]
]

START_DATE = datetime(2010, 1, 1)
END_DATE   = datetime(2024, 12, 31)
DATE_RANGE = int((END_DATE - START_DATE).total_seconds())


def random_datetime() -> datetime:
    """Return a random datetime between 2010-01-01 and 2024-12-31."""
    tz    = random.choice(TZ_OFFSETS)
    delta = random.randint(0, DATE_RANGE)
    dt    = START_DATE + timedelta(seconds=delta)
    return dt.replace(tzinfo=tz)


def format_goodreads_date(dt: datetime) -> str:
    """
    Format a datetime to the Goodreads string style.
    datetime.strftime %z gives '+0800'; Goodreads uses '+0800' → compatible.
    """
    return dt.strftime(GOODREADS_FMT)


def maybe_date_or_empty(probability: float = 0.70) -> str:
    """Return a formatted date string with `probability`, else empty string."""
    if random.random() < probability:
        return format_goodreads_date(random_datetime())
    return ""

# ─────────────────────────────────────────────
# 4. DISTRIBUTION LOGIC (Graph-friendly)
# ─────────────────────────────────────────────
# To simulate a realistic user-item bipartite graph:
#   - ~10% of users are "power users" → assigned ~40% of all interactions
#   - ~30% of users are "regular users" → assigned ~35% of interactions
#   - ~60% of users are "light users"  → remaining 25% of interactions
#
# Similarly, books follow a power-law: a few popular books get many ratings.

POWER_USER_CUTOFF   = int(NUM_USERS * 0.10)   # top 10%  → 50 users
REGULAR_USER_CUTOFF = int(NUM_USERS * 0.40)   # next 30% → 150 users
# remaining 60%                                           → 300 users

# Split interaction budget
POWER_INTERACTIONS   = int(TOTAL_INTERACTIONS * 0.40)  # 4,000
REGULAR_INTERACTIONS = int(TOTAL_INTERACTIONS * 0.35)  # 3,500
LIGHT_INTERACTIONS   = TOTAL_INTERACTIONS - POWER_INTERACTIONS - REGULAR_INTERACTIONS  # 2,500

power_users   = USER_POOL[:POWER_USER_CUTOFF]
regular_users = USER_POOL[POWER_USER_CUTOFF:REGULAR_USER_CUTOFF]
light_users   = USER_POOL[REGULAR_USER_CUTOFF:]

# Book popularity: top-20% books get ~60% of selections (Pareto-ish)
POPULAR_BOOK_CUTOFF = int(NUM_BOOKS * 0.20)   # 40 books
popular_books = BOOK_POOL[:POPULAR_BOOK_CUTOFF]
niche_books   = BOOK_POOL[POPULAR_BOOK_CUTOFF:]


def sample_user() -> str:
    """Sample a user according to the power-law distribution."""
    r = random.random()
    if r < 0.40:
        return random.choice(power_users)
    elif r < 0.75:
        return random.choice(regular_users)
    else:
        return random.choice(light_users)


def sample_book() -> str:
    """Sample a book with popularity bias (popular books appear more often)."""
    if random.random() < 0.60:
        return random.choice(popular_books)
    return random.choice(niche_books)

# ─────────────────────────────────────────────
# 5. SINGLE INTERACTION RECORD BUILDER
# ─────────────────────────────────────────────

def build_interaction() -> dict:
    """
    Construct one interaction record matching the Goodreads schema.
    """
    is_read   = random.random() < 0.75        # 75% chance the book was read
    date_added = format_goodreads_date(random_datetime())

    # date_updated: always present if the shelf was ever touched
    date_updated = format_goodreads_date(random_datetime())

    # read_at / started_at: only meaningful if the book was actually read
    if is_read:
        read_at    = maybe_date_or_empty(probability=0.85)
        started_at = maybe_date_or_empty(probability=0.70)
    else:
        read_at    = ""
        started_at = ""

    # Rating: 0 means "no rating given" (common for unread or quickly shelved)
    if is_read:
        # Slight positive bias (4s and 5s more common, mimicking real data)
        rating = random.choices(
            population=[0, 1, 2, 3, 4, 5],
            weights   =[5, 3, 7, 15, 35, 35],
        )[0]
    else:
        rating = 0   # unread → no meaningful rating

    return {
        "user_id"               : sample_user(),
        "book_id"               : sample_book(),
        "review_id"             : make_hex_id(),
        "is_read"               : is_read,
        "rating"                : rating,
        "review_text_incomplete": "",         # always empty in toy data
        "date_added"            : date_added,
        "date_updated"          : date_updated,
        "read_at"               : read_at,
        "started_at"            : started_at,
    }

# ─────────────────────────────────────────────
# 6. MAIN GENERATION LOOP
# ─────────────────────────────────────────────

def generate_dataset(n: int = TOTAL_INTERACTIONS) -> list[dict]:
    """Generate `n` interaction records."""
    print(f"⚙️  Generating {n:,} interactions ...")
    dataset = [build_interaction() for _ in range(n)]
    print(f"✅  Records generated.")
    return dataset


def save_jsonl(records: list[dict], path: str) -> None:
    """Write records as Line-delimited JSON (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"💾  Saved → {path}  ({len(records):,} lines)")


# ─────────────────────────────────────────────
# 7. STATISTICS REPORT
# ─────────────────────────────────────────────

def print_statistics(records: list[dict]) -> None:
    """Compute and display dataset statistics."""

    # Aggregate counters
    user_interactions  = defaultdict(int)
    book_interactions  = defaultdict(int)
    rating_dist        = defaultdict(int)
    is_read_count      = 0

    for rec in records:
        user_interactions[rec["user_id"]] += 1
        book_interactions[rec["book_id"]] += 1
        rating_dist[rec["rating"]]        += 1
        if rec["is_read"]:
            is_read_count += 1

    ui_values = sorted(user_interactions.values(), reverse=True)
    bi_values = sorted(book_interactions.values(), reverse=True)

    print("\n" + "═" * 52)
    print("  📊  DATASET STATISTICS")
    print("═" * 52)
    print(f"  Total interactions  : {len(records):>8,}")
    print(f"  Unique users        : {len(user_interactions):>8,}")
    print(f"  Unique books        : {len(book_interactions):>8,}")
    print(f"  Is-read (True)      : {is_read_count:>8,}  "
          f"({is_read_count/len(records)*100:.1f}%)")
    print()

    print("  📚  User interaction distribution")
    print(f"    Max per user      : {ui_values[0]:>6,}")
    print(f"    Median per user   : {ui_values[len(ui_values)//2]:>6,}")
    print(f"    Min per user      : {ui_values[-1]:>6,}")

    print()
    print("  📖  Book popularity distribution")
    print(f"    Max per book      : {bi_values[0]:>6,}")
    print(f"    Median per book   : {bi_values[len(bi_values)//2]:>6,}")
    print(f"    Min per book      : {bi_values[-1]:>6,}")

    print()
    print("  ⭐  Rating distribution")
    for star in range(6):
        cnt = rating_dist.get(star, 0)
        bar = "█" * (cnt // 50)
        print(f"    {star} stars : {cnt:>5,}  {bar}")

    print("═" * 52 + "\n")


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    records = generate_dataset(TOTAL_INTERACTIONS)
    save_jsonl(records, OUTPUT_FILE)
    print_statistics(records)
