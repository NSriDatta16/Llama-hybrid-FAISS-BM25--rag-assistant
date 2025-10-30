# backend/scripts/export_wikipedia_ai.py
"""
Export AI / Data Science content from Wikipedia using wikipedia-api (no deprecated HF scripts).
Writes one cleaned .txt per page into backend/data/wiki_ai_txt/.

Run:
  python -m backend.scripts.export_wikipedia_ai --limit 8000 --depth 1
or:
  python backend\scripts\export_wikipedia_ai.py --limit 8000 --depth 1
"""

import os, re, argparse, time, unicodedata, sys
from typing import Set, Dict
# --- allow running the script directly (python backend/scripts/..)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wikipediaapi
from backend.retrieval.hybrid import simple_clean, strip_urls, chunk_text  # reuse your cleaners

OUT_DIR = os.path.join("backend", "data", "wiki_ai_txt")

SEED_CATEGORIES = [
    "Category:Artificial_intelligence",
    "Category:Machine_learning",
    "Category:Deep_learning",
    "Category:Natural_language_processing",
    "Category:Computer_vision",
    "Category:Data_science",
    "Category:Reinforcement_learning",
    "Category:Big_data",
    "Category:Knowledge_representation",
    "Category:Expert_systems",
    "Category:Statistical_learning",
]

SLUG_RE = re.compile(r"[^a-z0-9\-]+")

def slugify(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace(" ", "-")
    s = SLUG_RE.sub("-", s).strip("-")
    return s or "untitled"

def save_text(title: str, text: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    fn = os.path.join(OUT_DIR, f"{slugify(title)}.txt")
    with open(fn, "w", encoding="utf-8") as f:
        f.write(text)

def main(limit: int, depth: int, min_chars: int, throttle: float):
    print("🔍 Crawling Wikipedia via wikipedia-api…")
    wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="FB_RAG_AI_DataScience_Crawler/1.0 (https://github.com/yourname/FB_RAG)")

    to_visit: Set[str] = set(SEED_CATEGORIES)
    visited: Set[str] = set()
    saved = 0

    while to_visit and saved < limit:
        cat_title = to_visit.pop()
        if cat_title in visited:
            continue
        visited.add(cat_title)

        cat = wiki.page(cat_title)
        if not cat.exists():
            continue

        # Traverse members (pages + subcategories)
        members: Dict[str, wikipediaapi.WikipediaPage] = cat.categorymembers
        for title, page in members.items():
            if saved >= limit:
                break

            if page.ns == wikipediaapi.Namespace.MAIN and page.text:
                # Clean + light filtering
                text = strip_urls(simple_clean(page.text))
                if len(text) < min_chars:
                    continue
                # Optional: pre-chunk to keep files smaller
                chunks = chunk_text(text, size=4000, overlap=0)  # big pages split into a few files
                if not chunks:
                    continue
                for i, ck in enumerate(chunks):
                    ttitle = title if i == 0 else f"{title} (part {i+1})"
                    save_text(ttitle, ck)
                    saved += 1
                    if saved % 100 == 0:
                        print(f" _saved {saved} / {limit}…")
                    if saved >= limit:
                        break
                time.sleep(throttle)

            # Follow subcategories up to `depth`
            if page.ns == wikipediaapi.Namespace.CATEGORY:
                if depth > 0:
                    to_visit.add(page.title)

        # Decrease depth after finishing a category layer
        if depth > 0:
            depth -= 1

    print(f"✅ Done. Wrote {saved} files to {OUT_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8000, help="Max number of text files to write")
    ap.add_argument("--depth", type=int, default=1, help="How many category levels to follow")
    ap.add_argument("--min_chars", type=int, default=800, help="Skip tiny pages")
    ap.add_argument("--throttle", type=float, default=0.1, help="Sleep seconds between page fetches")
    args = ap.parse_args()
    main(args.limit, args.depth, args.min_chars, args.throttle)
