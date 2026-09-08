import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CSV_PATH = Path("out/embedding_analysis/top_10000_by_revenue.csv")
OUT_DIR = Path("out/embedding_analysis/vectors")
MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
DEVICE = "mps"
BATCH_SIZE = 64
# Starting point for --batch-size auto -- gets grown/shrunk from here, so
# it doesn't need to be exact, just a reasonable first guess.
AUTO_BATCH_START = 64
AUTO_BATCH_MIN = 1
AUTO_BATCH_GROWTH = 1.2
# Consecutive successful chunks before trying a bigger batch size again --
# avoids growing back into the same OOM every other chunk.
AUTO_BATCH_GROW_AFTER = 3


def parse_batch_size(value):
    if value == "auto":
        return value
    return int(value)


def _is_oom_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _empty_device_cache(device):
    device = str(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def encode_adaptive(model, texts, start_batch_size=AUTO_BATCH_START, device=DEVICE):
    # Sort longest-first: the biggest texts are the ones that risk OOM, so
    # we find a batch size that survives them immediately instead of
    # discovering it after already embedding most of the dataset. Once a
    # batch size survives the longest texts, it's safe (or can grow) for
    # everything shorter that follows.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]), reverse=True)
    sorted_texts = [texts[i] for i in order]

    embeddings_by_original_idx = {}
    batch_size = max(start_batch_size, AUTO_BATCH_MIN)
    consecutive_successes = 0
    pos = 0
    pbar = tqdm(total=len(sorted_texts), desc="Encoding (auto batch)")

    while pos < len(sorted_texts):
        chunk_indices = order[pos:pos + batch_size]
        chunk_texts = sorted_texts[pos:pos + batch_size]
        try:
            chunk_embeddings = model.encode(
                chunk_texts,
                batch_size=len(chunk_texts),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except RuntimeError as exc:
            if not _is_oom_error(exc) or batch_size <= AUTO_BATCH_MIN:
                raise
            _empty_device_cache(device)
            batch_size = max(AUTO_BATCH_MIN, batch_size // 2)
            consecutive_successes = 0
            pbar.set_postfix(batch_size=batch_size, event="oom, backing off")
            continue

        for idx, embedding in zip(chunk_indices, chunk_embeddings):
            embeddings_by_original_idx[idx] = embedding

        pos += len(chunk_texts)
        pbar.update(len(chunk_texts))
        consecutive_successes += 1
        if consecutive_successes >= AUTO_BATCH_GROW_AFTER:
            consecutive_successes = 0
            batch_size = max(AUTO_BATCH_MIN, round(batch_size * AUTO_BATCH_GROWTH))
        pbar.set_postfix(batch_size=batch_size)

    pbar.close()
    return np.stack([embeddings_by_original_idx[i] for i in range(len(texts))])


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main(
    csv_path=CSV_PATH,
    out_dir=OUT_DIR,
    model_name=MODEL_NAME,
    device=DEVICE,
    batch_size=BATCH_SIZE,
    tag=None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # subset.py already builds embedding_text into the CSV -- no need to
    # reassemble it here.
    df = pd.read_csv(csv_path)
    tag = tag or Path(csv_path).stem

    print(f"Loading model: {model_name} (device={device})")
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"dtype": "auto"},
    )

    texts = df["embedding_text"].fillna("").tolist()
    if batch_size == "auto":
        embeddings = encode_adaptive(model, texts, device=device)
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    assert embeddings.shape[0] == len(df)

    print("Device:", model.device)
    print("Embeddings:", embeddings.shape)

    slug = slugify(model_name)
    vectors_path = out_dir / f"{slug}_vectors_{tag}.npy"
    ids_path = out_dir / f"{slug}_movie_ids_{tag}.npy"

    np.save(vectors_path, embeddings.astype(np.float32))
    np.save(ids_path, df["id"].to_numpy())
    print(f"Saved vectors -> {vectors_path}")
    print(f"Saved movie ids -> {ids_path}")
    return vectors_path, ids_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode embedding_text with a sentence-transformers model")
    parser.add_argument("csv_path", nargs="?", default=str(CSV_PATH),
                         help="subset.py output CSV with an embedding_text column")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--model-name", default=MODEL_NAME,
                         help="Any sentence-transformers model id or local path")
    parser.add_argument("--device", default=DEVICE, help="e.g. mps, cuda, cpu")
    parser.add_argument("--batch-size", type=parse_batch_size, default=BATCH_SIZE,
                         help="Fixed batch size, or 'auto' to grow/shrink around OOMs "
                              "and fit as much as the device can handle")
    parser.add_argument("--tag", default=None,
                         help="Suffix for output filenames (default: CSV filename stem, "
                              "e.g. top_10000_by_revenue)")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        tag=args.tag,
    )
