import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

CSV_PATH = Path("out/embedding_analysis/top_10000_by_revenue.csv")
OUT_DIR = Path("out/embedding_analysis/vectors")
MODEL_NAME = "microsoft/harrier-oss-v1-0.6b"
DEVICE = "mps"
BATCH_SIZE = 64


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

    embeddings = model.encode(
        df["embedding_text"].fillna("").tolist(),
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
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
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
