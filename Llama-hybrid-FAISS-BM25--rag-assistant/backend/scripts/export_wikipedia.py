# backend/scripts/export_wikipedia.py
import os, uuid, argparse
from datasets import load_dataset, IterableDataset
from typing import Iterable, Dict, Any

def _export_iterable(ds_iter: Iterable[Dict[str, Any]], out_dir: str, limit: int) -> int:
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for ex in ds_iter:
        text = (ex.get("text") or "").strip()
        if not text:
            continue
        fn = os.path.join(out_dir, f"{uuid.uuid4().hex}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(text)
        count += 1
        if count % 1000 == 0:
            print(f"… exported {count} files")
        if limit and count >= limit:
            break
    return count

def export_wiki_like(out_dir: str, subset: str, limit: int) -> int:
    """
    Preferred: wikimedia/wikipedia with trust_remote_code=True (new official source).
    Falls back to wikitext / squad if unavailable.
    """
    tried = []

    # 1) wikimedia/wikipedia (new) — try a few snapshots
    configs = [subset] if subset else ["20240501.en", "20231101.en", "20230501.en"]
    for cfg in configs:
        try:
            print(f"⏳ Loading wikimedia/wikipedia ({cfg}) streaming…")
            ds = load_dataset(
                "wikimedia/wikipedia",
                cfg,
                split="train",
                streaming=True,            # stream rows, no big local download
                trust_remote_code=True     # REQUIRED after the change you hit
            )
            n = _export_iterable(ds, out_dir, limit)
            print(f"✅ Exported {n} docs from wikimedia/wikipedia:{cfg}")
            return n
        except Exception as e:
            tried.append(("wikimedia/wikipedia", cfg, str(e)))
            print(f"⚠️ Failed wikimedia/wikipedia:{cfg}: {e}")

    # 2) Wikitext (stable small corpus)
    try:
        print("⏳ Loading wikitext-103-raw-v1 …")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        n = 0
        for ex in ds:
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            fn = os.path.join(out_dir, f"{uuid.uuid4().hex}.txt")
            with open(fn, "w", encoding="utf-8") as f:
                f.write(text)
            n += 1
            if limit and n >= limit:
                break
            if n % 1000 == 0:
                print(f"… exported {n} files")
        print(f"✅ Exported {n} docs from wikitext-103-raw-v1")
        return n
    except Exception as e:
        tried.append(("wikitext-103-raw-v1", "", str(e)))
        print(f"⚠️ Failed wikitext: {e}")

    # 3) SQuAD v2 contexts (very clean English)
    try:
        print("⏳ Loading squad_v2 …")
        ds = load_dataset("squad_v2")
        out = 0
        for split in ["train", "validation"]:
            for ex in ds[split]:
                text = (ex.get("context") or "").strip()
                if not text:
                    continue
                fn = os.path.join(out_dir, f"{uuid.uuid4().hex}.txt")
                with open(fn, "w", encoding="utf-8") as f:
                    f.write(text)
                out += 1
                if limit and out >= limit:
                    break
                if out % 1000 == 0:
                    print(f"… exported {out} files")
            if limit and out >= limit:
                break
        print(f"✅ Exported {out} docs from SQuAD v2")
        return out
    except Exception as e:
        tried.append(("squad_v2", "", str(e)))
        print(f"⚠️ Failed SQuAD: {e}")

    raise RuntimeError(f"Could not export any dataset. Tried: {tried}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="backend/data/wiki_export")
    ap.add_argument("--subset", default="", help="e.g. 20240501.en for wikimedia/wikipedia")
    ap.add_argument("--limit", type=int, default=20000)
    args = ap.parse_args()
    n = export_wiki_like(args.out_dir, args.subset, args.limit)
    print(f"✅ Export complete: {n} files -> {args.out_dir}")
