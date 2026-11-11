from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sftl54_pytorch_dataset import SFTL54PairDataset
from sftl54_transformer_adapter import EdgeFaceTransformerAdapter
from sftl54_xedgeface_train import load_pretrained_edgeface
from regional_feature_extractor import REGION_NAMES


ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "sftl54_uncertainty_visualizations")
    parser.add_argument("--checkpoint", type=Path, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgeFaceTransformerAdapter(copy.deepcopy(load_pretrained_edgeface())).to(device)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    dataset = SFTL54PairDataset(split="val")
    loader = DataLoader(dataset, batch_size=min(8, args.count), shuffle=False)
    rows = []
    with torch.inference_mode():
        for batch in loader:
            output = model.forward_with_uncertainty(
                batch["rgb"].to(device), batch["thermal"].to(device), batch["landmarks"].to(device)
            )
            for i, pair_key in enumerate(batch["pair_key"]):
                if len({r["pair_key"] for r in rows}) >= args.count:
                    break
                for region_index, region in enumerate(REGION_NAMES):
                    rows.append({
                        "pair_key": pair_key,
                        "subject_id": int(batch["subject_id"][i]),
                        "region": region,
                        "rgb_log_uncertainty": float(output["rgb_uncertainty"][i, region_index]),
                        "thermal_log_uncertainty": float(output["thermal_uncertainty"][i, region_index]),
                        "combined_log_uncertainty": float(output["combined_uncertainty"][i, region_index]),
                        "reliability": float(output["regional_reliability"][i, region_index]),
                    })
            if len({r["pair_key"] for r in rows}) >= args.count:
                break
    report = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "uncertainty.csv", index=False)
    for pair_key, group in report.groupby("pair_key", sort=False):
        figure, axis = plt.subplots(figsize=(7, 3))
        x = range(len(REGION_NAMES))
        axis.bar([v - 0.25 for v in x], group["rgb_log_uncertainty"], width=0.25, label="RGB")
        axis.bar(x, group["thermal_log_uncertainty"], width=0.25, label="Thermal")
        axis.bar([v + 0.25 for v in x], group["reliability"], width=0.25, label="Reliability")
        axis.set_xticks(list(x), REGION_NAMES)
        axis.set_title(str(pair_key))
        axis.legend()
        figure.tight_layout()
        figure.savefig(args.output_dir / f"{pair_key}.png", dpi=150)
        plt.close(figure)
    print(f"saved {report['pair_key'].nunique()} visualizations and {len(report)} rows")


if __name__ == "__main__":
    main()
