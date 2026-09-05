from pathlib import Path

from datasets import Dataset
from huggingface_hub import hf_hub_download

DATA_DIR = Path(__file__).resolve().parent
REPO_ID = "seamew/ChnSentiCorp"

FILES = {
    "train": "chn_senti_corp-train.arrow",
    "validation": "chn_senti_corp-validation.arrow",
    "test": "chn_senti_corp-test.arrow",
}

for split, filename in FILES.items():
    print(f"正在下载 {split}: {filename}")

    arrow_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )

    ds = Dataset.from_file(arrow_path)

    out_path = DATA_DIR / f"{split}.parquet"
    ds.to_parquet(str(out_path))

    print(f"{split}: {len(ds)} 条 -> {out_path}")

print("\n下载完成。")
