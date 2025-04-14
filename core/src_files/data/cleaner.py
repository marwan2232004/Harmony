import pandas as pd
from pathlib import Path
# CONFIG (Edit these)
TSV_PATH = "your_data.tsv"  # Path to your TSV
AUDIO_DIR = "dataset"  # Folder with MP3s
OUTPUT_CSV = "cleaned_data.csv"  # Output file
MIN_NET_VOTES = 2  # (up_votes - down_votes) >= 2


def clean_data():
    # 1. Load TSV with only needed columns
    df = pd.read_csv(
        TSV_PATH,
        sep='\t',
        usecols=['path', 'gender', 'up_vote', 'down_vote']
    )

    # 2. Filter by vote quality
    df = df[(df['up_vote'] - df['down_vote']) >= MIN_NET_VOTES].copy()

    # 3. Check file existence (exact filename match in AUDIO_DIR)
    df['exists'] = df['path'].apply(
        lambda x: (Path(AUDIO_DIR) / x).exists()
    )

    # 4. Remove rows with missing files
    clean_df = df[df['exists']].copy()

    # 5. Delete audio files not in filtered TSV
    valid_files = set(clean_df['path'])
    for mp3 in Path(AUDIO_DIR).glob("*.mp3"):
        if mp3.name not in valid_files:
            mp3.unlink()  # Delete orphaned file
            print(f"Deleted: {mp3.name}")

    # 6. Save clean dataset (relative paths)
    clean_df[['path', 'gender']].to_csv(OUTPUT_CSV, index=False)
    print(f"- Removed due to low votes: {len(df) - len(df[df['exists']])}")
    print(f"- Removed missing files: {len(df[df['exists']]) - len(clean_df)}")
    print(f"- Final samples: {len(clean_df)}")
    print(f"Saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    clean_data()
