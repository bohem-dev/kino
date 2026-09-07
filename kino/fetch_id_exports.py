import datetime
import gzip
import os
import shutil

import pandas as pd
import requests
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words("english"))
pd.set_option("display.max_colwidth", None)

DOWNLOAD_DIR = "downloads/tmdb-ids"
BASE_URL = "https://files.tmdb.org/p/exports"

# All exports TMDB publishes daily
EXPORT_TYPES = {
    "movie_ids": "movie_ids",
    "tv_series_ids": "tv_series_ids",
    "person_ids": "person_ids",
    "collection_ids": "collection_ids",
    "tv_network_ids": "tv_network_ids",
    "keyword_ids": "keyword_ids",
    "production_company_ids": "production_company_ids",
    "adult_movie_ids": "adult_movie_ids",
    "adult_tv_series_ids": "adult_tv_series_ids",
    "adult_person_ids": "adult_person_ids",
}


def get_export_date():
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_utc >= cutoff:
        return now_utc.date()
    return (now_utc - datetime.timedelta(days=1)).date()


def download_all(export_prefixes=None, date=None, extract=True):
    date = date or get_export_date()
    export_prefixes = export_prefixes or list(EXPORT_TYPES.values())

    def build_filename(export_prefix, date):
        date_str = date.strftime("%m_%d_%Y")
        return f"{export_prefix}_{date_str}.json.gz"

    def download_export(export_prefix, date, download_dir=DOWNLOAD_DIR, extract=True):
        os.makedirs(download_dir, exist_ok=True)

        filename = build_filename(export_prefix, date)
        local_gz_path = os.path.join(download_dir, filename)
        local_json_path = local_gz_path[:-3]  # strip .gz

        # Skip if we already have this export (gz or extracted json)
        if os.path.exists(local_gz_path) or os.path.exists(local_json_path):
            print(f"[skip] {filename} already present.")
            return local_json_path if os.path.exists(local_json_path) else local_gz_path

        url = f"{BASE_URL}/{filename}"
        print(f"[fetch] {url}")

        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 404:
            print(f"[warn] {filename} not found yet (404). It may not be published for this date/type.")
            return None
        resp.raise_for_status()

        with open(local_gz_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[ok] saved {local_gz_path}")

        if extract:
            with gzip.open(local_gz_path, "rb") as f_in, open(local_json_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(local_gz_path)
            print(f"[ok] extracted -> {local_json_path}")
            return local_json_path

        return local_gz_path

    results = {}
    for prefix in export_prefixes:
        path = download_export(prefix, date, extract=extract)
        results[prefix] = path
    return results


def main():
    date = get_export_date()
    print(f"Using export date: {date}")

    result = download_all(date=date)
    print(result)

    out_dir = "out"
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    yesterday = date - datetime.timedelta(days=1)
    report = []
    summary_rows = []

    for prefix in EXPORT_TYPES.values():
        today_path = os.path.join(DOWNLOAD_DIR, f"{prefix}_{date.strftime('%m_%d_%Y')}.json")
        yesterday_path = os.path.join(DOWNLOAD_DIR, f"{prefix}_{yesterday.strftime('%m_%d_%Y')}.json")

        print(f"[processing] [a] -> {prefix}_{date.strftime('%m_%d_%Y')}")
        print(f"[processing] [b] -> {prefix}_{yesterday.strftime('%m_%d_%Y')}")

        if not os.path.exists(today_path):
            continue

        df = pd.read_json(today_path, lines=True)
        name_col = "name" if "name" in df.columns else "original_title" if "original_title" in df.columns else "original_name"
        df["snapshot_date"] = date

        df.to_csv(os.path.join(data_dir, f"{prefix}_snapshot_{date.strftime('%m_%d_%Y')}.csv"), index=False)

        report.append(f"\n{'='*70}\n{prefix.upper()} — {date}\n{'='*70}")
        report.append(f"Total entries: {len(df):,}")

        if "popularity" in df.columns:
            pop = df["popularity"]
            report.append(f"Popularity — mean: {pop.mean():.4f}, median: {pop.median():.4f}, std: {pop.std():.4f}, min: {pop.min():.4f}, max: {pop.max():.4f}")
            report.append(f"Zero popularity entries: {(pop == 0).sum():,} ({(pop == 0).mean()*100:.1f}%)")
            report.append(f"Percentiles:\n{pop.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_string()}")
            report.append(f"Top 10 most popular:\n{df.nlargest(10, 'popularity')[['id', name_col, 'popularity']].to_string(index=False)}")
            report.append(f"Bottom 10 (nonzero) popularity:\n{df[pop > 0].nsmallest(10, 'popularity')[['id', name_col, 'popularity']].to_string(index=False)}")

        if "adult" in df.columns:
            report.append(f"Adult flagged: {df['adult'].sum():,} ({df['adult'].mean()*100:.2f}%)")
        if "video" in df.columns:
            report.append(f"Video (direct-to-video) flagged: {df['video'].sum():,} ({df['video'].mean()*100:.2f}%)")

        # lexical analysis of the title/name field — words, lengths, vocabulary, everything
        titles = df[name_col].dropna().astype(str)
        tokens = [w.lower() for t in titles for w in word_tokenize(t) if w.isalpha()]
        word_counts = Counter(w for w in tokens if w not in STOPWORDS)
        char_lengths = titles.str.len()
        word_lengths = titles.str.split().str.len()

        report.append(f"\n--- Lexical analysis ({name_col}) ---")
        report.append(f"Total word tokens: {len(tokens):,}, unique words (stopwords removed): {len(word_counts):,}")
        report.append(f"Title length (chars) — mean: {char_lengths.mean():.1f}, median: {char_lengths.median():.1f}, max: {char_lengths.max()}")
        report.append(f"Title length (words) — mean: {word_lengths.mean():.1f}, median: {word_lengths.median():.1f}, max: {word_lengths.max()}")
        report.append(f"Most common words (full list in word_freq CSV):\n{pd.Series(dict(word_counts.most_common(25)), name='count').to_string()}")
        report.append(f"Longest titles:\n{titles.loc[char_lengths.nlargest(10).index].to_string(index=False)}")
        report.append(f"Shortest (nonzero) titles:\n{titles.loc[char_lengths[char_lengths > 0].nsmallest(10).index].to_string(index=False)}")

        word_freq = pd.DataFrame(word_counts.most_common(), columns=["word", "count"])
        word_freq.insert(0, "snapshot_date", date)
        word_freq.insert(1, "export_type", prefix)
        word_freq.to_csv(os.path.join(data_dir, f"{prefix}_word_freq_{date.strftime('%m_%d_%Y')}.csv"), index=False)

        summary_row = {
            "date": date, "export_type": prefix, "total_today": len(df),
            "vocab_size": len(word_counts), "top_word": word_counts.most_common(1)[0][0] if word_counts else None,
            "avg_title_length_chars": char_lengths.mean(), "avg_title_length_words": word_lengths.mean(),
        }

        if os.path.exists(yesterday_path):
            df_y = pd.read_json(yesterday_path, lines=True)
            new_ids = set(df["id"]) - set(df_y["id"])
            removed_ids = set(df_y["id"]) - set(df["id"])

            report.append(f"\n--- Change vs {yesterday} ---")
            report.append(f"Count change: {len(df_y):,} -> {len(df):,} ({len(df)-len(df_y):+,})")
            report.append(f"New entries: {len(new_ids):,}")
            report.append(f"Removed entries: {len(removed_ids):,}")

            if new_ids:
                new_rows = df[df["id"].isin(new_ids)]
                if "popularity" in new_rows.columns:
                    new_rows = new_rows.sort_values("popularity", ascending=False)
                cols = [c for c in ["id", name_col, "popularity"] if c in new_rows.columns]
                report.append(f"Sample new entries (see full CSV for all {len(new_ids):,}):\n{new_rows[cols].head(10).to_string(index=False)}")

            # full outer-join diff, every row not just a top-N sample, saved for later lookup
            diff = df_y.merge(df, on="id", how="outer", suffixes=("_yesterday", "_today"), indicator=True)
            diff["status"] = diff["_merge"].map({"left_only": "removed", "right_only": "new", "both": "existing"})
            diff = diff.drop(columns="_merge")
            if "popularity" in df.columns:
                diff["popularity_change"] = diff["popularity_today"] - diff["popularity_yesterday"]
            diff.insert(0, "compared_date", date)
            diff.insert(1, "compared_against", yesterday)
            diff.to_csv(os.path.join(data_dir, f"{prefix}_diff_{date.strftime('%m_%d_%Y')}.csv"), index=False)

            summary_row.update({
                "compared_against": yesterday, "total_yesterday": len(df_y),
                "new_entries": len(new_ids), "removed_entries": len(removed_ids),
            })

            if "popularity" in df.columns:
                merged = df_y[["id", "popularity"]].merge(df[["id", "popularity"]], on="id", suffixes=("_yesterday", "_today"))
                merged["change"] = merged["popularity_today"] - merged["popularity_yesterday"]
                merged = merged.merge(df[["id", name_col]], on="id")
                report.append(f"Global popularity shift (mean change): {merged['change'].mean():+.5f}")
                report.append(f"Top popularity gainers (full ranking in diff CSV):\n{merged.sort_values('change', ascending=False).head(10)[[name_col, 'popularity_yesterday', 'popularity_today', 'change']].to_string(index=False)}")
                report.append(f"Top popularity losers (full ranking in diff CSV):\n{merged.sort_values('change').head(10)[[name_col, 'popularity_yesterday', 'popularity_today', 'change']].to_string(index=False)}")
                summary_row["popularity_mean_change"] = merged["change"].mean()
                summary_row["popularity_mean_today"] = df["popularity"].mean()
                summary_row["popularity_max_today"] = df["popularity"].max()

        summary_rows.append(summary_row)

    out_path = os.path.join(out_dir, f"report_{date.strftime('%m_%d_%Y')}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(str(x) for x in report))

    print(f"[ok] wrote report -> {out_path}")

    # running longitudinal log, appended to (not overwritten) so history builds up day over day
    if summary_rows:
        history_path = os.path.join(data_dir, "summary_history.csv")
        history = pd.DataFrame(summary_rows)
        history.to_csv(history_path, mode="a", index=False, header=not os.path.exists(history_path))
        print(f"[ok] appended {len(history)} rows -> {history_path}")


if __name__ == "__main__":
    main()
