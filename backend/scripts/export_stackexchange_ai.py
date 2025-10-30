# backend/scripts/export_stackexchange_ai.py
# Extract AI/DS posts from Stack Exchange dumps into backend/data/stackexchange_ai_txt/
# Usage:
#   python -m backend.scripts.export_stackexchange_ai ^
#     --dump_dir "C:\\Users\\srida\\OneDrive\\Downloads\\SE_DUMPS" ^
#     --sites datascience crossvalidated stackoverflow ^
#     --max_docs 200000


import os, re, argparse, html, xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

OUT_DIR = Path("backend/data/stackexchange_ai_txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tag names to match (normalized, without angle brackets)
TAG_WHITELIST = {
    # Core
    "machine-learning", "deep-learning", "neural-network", "convolutional-neural-network",
    "recurrent-neural-network", "transformer", "attention", "autoencoder", "gan",
    "reinforcement-learning", "markov-decision-process", "q-learning", "policy-gradient",
    "nlp", "natural-language-processing", "llm", "large-language-model", "bert", "gpt",
    "rag", "retrieval-augmented-generation", "vector-search", "embeddings",
    "agents", "multi-agent", "prompt-engineering",
    # DS/Stats
    "data-science", "statistics", "feature-selection", "dimensionality-reduction",
    "pca", "svm", "random-forest", "xgboost", "gradient-boosting", "logistic-regression",
    "time-series", "bayesian", "markov-chain", "mcmc",
}

# Keyword fallback (case-insensitive, plain text of Title+Body)
KEYWORDS = [
    "machine learning", "deep learning", "neural network", "cnn", "rnn",
    "transformer", "attention", "autoencoder", "vae", "gan", "diffusion model",
    "reinforcement learning", "q-learning", "policy gradient", "mdp",
    "nlp", "language model", "bert", "gpt", "rag", "vector database",
    "embedding", "retrieval", "agent", "multi-agent",
    "data science", "feature engineering", "dimensionality reduction",
    "pca", "svm", "random forest", "xgboost", "logistic regression",
    "bayesian", "time series", "markov chain", "mcmc",
]

KW_RE = re.compile("|".join([re.escape(k) for k in KEYWORDS]), re.IGNORECASE)

TAG_RE = re.compile(r"<([^>]+)>")  # parses <tag1><tag2>… format
HTML_TAG_RE = re.compile(r"<[^>]+>")  # strip HTML tags from Body

def norm_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = HTML_TAG_RE.sub(" ", s)  # remove HTML tags
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tag_list(tags_field: str) -> Iterable[str]:
    if not tags_field:
        return []
    # Posts.xml stores tags like "<machine-learning><python>"
    return [t.lower() for t in TAG_RE.findall(tags_field)]

def looks_ai(tags: Iterable[str], text: str) -> bool:
    # tag match OR keyword match
    if any(t in TAG_WHITELIST for t in tags):
        return True
    return bool(KW_RE.search(text))

def export_site(posts_xml: Path, site_name: str, max_docs: int, min_chars: int) -> int:
    if not posts_xml.exists():
        print(f"⚠️  Missing: {posts_xml}")
        return 0
    print(f"🔍 Parsing {posts_xml} … (this can take a while)")

    count = 0
    # iterparse to keep memory low
    for event, elem in ET.iterparse(str(posts_xml), events=("end",)):
        if elem.tag != "row":
            continue

        title = elem.attrib.get("Title", "")
        body = elem.attrib.get("Body", "")
        tags_field = elem.attrib.get("Tags", "")
        post_id = elem.attrib.get("Id", "")
        parent_id = elem.attrib.get("ParentId")  # answers have ParentId

        # Normalize text
        title_n = norm_text(title)
        body_n = norm_text(body)
        text = (title_n + "\n\n" + body_n).strip()
        tags = list(tag_list(tags_field))

        if len(text) < min_chars:
            elem.clear()
            continue

        if looks_ai(tags, text):
            # Include some lightweight header for context
            header = [
                f"[site]: {site_name}",
                f"[post_id]: {post_id}",
                f"[parent_id]: {parent_id or ''}",
                f"[tags]: {', '.join(tags)}",
                "",
            ]
            out = "\n".join(header) + text + "\n"
            out_path = OUT_DIR / f"{site_name}_{post_id}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(out)
            count += 1
            if count % 500 == 0:
                print(f"  … saved {count}")

            if max_docs and count >= max_docs:
                break

        elem.clear()
    return count

def main(dump_dir: str, sites, max_docs: int, min_chars: int):
    total = 0
    for site in sites:
        posts_xml = Path(dump_dir) / site / "Posts.xml"
        total += export_site(posts_xml, site, max_docs, min_chars)
    print(f"✅ Saved {total} files to {OUT_DIR}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True, help="Folder containing site folders (each with Posts.xml)")
    ap.add_argument("--sites", nargs="+", default=["datascience", "crossvalidated", "stackoverflow"])
    ap.add_argument("--max_docs", type=int, default=200000)
    ap.add_argument("--min_chars", type=int, default=200)  # be generous
    args = ap.parse_args()
    main(args.dump_dir, args.sites, args.max_docs, args.min_chars)
