import argparse
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_COMMIT = "3e8fc4d69fbff2c9060bdbb347f2bd94847f75ea"
ARCHIVE_URL = (
    "https://codeload.github.com/xhwang22/HallucinationDetection/zip/"
    + SOURCE_COMMIT
)
SOURCE_REPO = "https://github.com/xhwang22/HallucinationDetection"


def main():
    parser = argparse.ArgumentParser(
        description="Download the released Wang et al. EMNLP 2023 experimental artifacts."
    )
    parser.add_argument("--output", default="data/wang")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        if not args.force:
            print(f"{output} already exists. Use --force to replace it.")
            return
        shutil.rmtree(output)

    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        zip_path = tmp / "wang.zip"
        print(f"Downloading Wang et al. released data at commit {SOURCE_COMMIT}...")
        urllib.request.urlretrieve(ARCHIVE_URL, zip_path)

        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path) as zf:
            top_level = f"HallucinationDetection-{SOURCE_COMMIT}/"
            prefix = top_level + "dataset/"
            members = [name for name in zf.namelist() if name.startswith(prefix)]
            zf.extractall(tmp, members=members)

        extracted_dataset = tmp / f"HallucinationDetection-{SOURCE_COMMIT}" / "dataset"
        if not extracted_dataset.exists():
            raise RuntimeError("Downloaded archive did not contain the expected dataset directory")
        shutil.copytree(extracted_dataset, output)

    metadata = {
        "source_repository": SOURCE_REPO,
        "source_commit": SOURCE_COMMIT,
        "source_archive": ARCHIVE_URL,
        "paper": "Hallucination Detection for Generative Large Language Models by Bayesian Sequential Estimation",
        "authors": "Wang et al.",
        "venue": "EMNLP 2023",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Data are the authors' released experimental artifacts. See their repository/paper for licensing and citation.",
    }
    with (output / "SOURCE.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Prepared Wang et al. data at {output}")
    print("Next: python main.py --smoke-test")


if __name__ == "__main__":
    main()
