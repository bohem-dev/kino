import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

CSV_PATH = Path("out/embedding_analysis/top_10000_by_revenue.csv")
OUT_DIR = Path("out/embedding_analysis/vectors")
MODEL_DIR = Path("out/embedding_analysis/models")

VECTOR_SIZE = 300
WINDOW = 8
MIN_COUNT = 3
EPOCHS = 20
WORKERS = os.cpu_count() or 1


def tokenize(text):
    return simple_preprocess(text, deacc=True)


def movie_vector(tokens, model):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)


def main(
    csv_path=CSV_PATH,
    out_dir=OUT_DIR,
    model_dir=MODEL_DIR,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    epochs=EPOCHS,
    workers=WORKERS,
    tag="10k",
):
    out_dir = Path(out_dir)
    model_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    corpus = [tokenize(t) for t in df["embedding_text"].fillna("").tolist()]

    print(f"Training Word2Vec on {len(corpus)} documents "
          f"(vector_size={vector_size}, window={window}, min_count={min_count}, epochs={epochs})...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        sg=1,  # skip-gram (better for the smaller, rarer-word-heavy corpus here than CBOW)
    )
    print(f"Vocabulary size: {len(model.wv)}")

    embeddings = np.stack([movie_vector(tokens, model) for tokens in corpus])
    assert embeddings.shape == (len(df), vector_size)
    print("Embeddings:", embeddings.shape)

    vectors_path = out_dir / f"word2vec_{vector_size}d_vectors_{tag}.npy"
    ids_path = out_dir / f"word2vec_{vector_size}d_movie_ids_{tag}.npy"
    np.save(vectors_path, embeddings)
    np.save(ids_path, df["id"].to_numpy())
    model.save(str(model_dir / f"word2vec_{vector_size}d_{tag}.model"))
    print(f"Saved vectors -> {vectors_path}")
    print(f"Saved movie ids -> {ids_path}")
    return vectors_path, ids_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Word2Vec model on embedding_text, mean-pool per movie")
    parser.add_argument("csv_path", nargs="?", default=str(CSV_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--min-count", type=int, default=MIN_COUNT)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--tag", default="10k",
                         help="Suffix for output filenames (match the --top-n used in subset.py)")
    args = parser.parse_args()

    main(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        model_dir=args.model_dir,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        tag=args.tag,
    )
