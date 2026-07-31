"""E0-A synthetic import/construction/forward smoke only.

This helper never loads SDWPF, trains, downloads, or writes formal result
artifacts. It writes one JSON file beside itself.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).with_name("audit_smoke_results.json")
TSLIB = ROOT / "Time-Series-Library"
sys.path.insert(0, str(TSLIB))


def args() -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        features="M",
        seq_len=144,
        label_len=48,
        pred_len=10,
        enc_in=16,
        dec_in=16,
        c_out=16,
        d_model=16,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=32,
        dropout=0.0,
        factor=1,
        activation="gelu",
        embed="timeF",
        freq="h",
        distil=True,
        top_k=3,
        num_kernels=3,
        moving_avg=25,
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method="avg",
        seg_len=12,
        patch_len=16,
        alpha=0.1,
        top_p=0.5,
        pos=1,
        individual=False,
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        device=torch.device("cpu"),
        batch_size=2,
        num_class=2,
        version="Fourier",
        modes=8,
        mode_select="random",
        output_attention=False,
        use_amp=False,
        use_multi_gpu=False,
        gpu=0,
        gpu_type="cuda",
        p_hidden_dims=[16, 16],
        p_hidden_layers=2,
        expand=2,
        d_conv=4,
        tv_dt=0,
        tv_B=0,
        tv_C=0,
        use_D=0,
    )


def run_one(name: str) -> dict:
    try:
        import importlib

        module = importlib.import_module(f"models.{name}")
        model = module.Model(args()).eval()
        x = torch.randn(2, 144, 16)
        mark = torch.zeros(2, 144, 4)
        dec = torch.zeros(2, 58, 16)
        dec_mark = torch.zeros(2, 58, 4)
        with torch.inference_mode():
            output = model(x, mark, dec, dec_mark)
        shape = list(output.shape) if hasattr(output, "shape") else str(type(output))
        return {"model": name, "status": "SMOKE_OK", "output_shape": shape}
    except Exception as exc:  # evidence collection intentionally continues
        return {
            "model": name,
            "status": "SMOKE_ERROR",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }


def main() -> None:
    torch.set_num_threads(1)
    names = [
        "DLinear", "LightTS", "TiDE", "SegRNN", "Transformer", "PatchTST",
        "iTransformer", "TimeXer", "TimesNet", "MICN", "WPMixer",
        "MultiPatchFormer", "TimeMixer", "TSMixer", "FreTS", "Crossformer",
        "MSGNet", "TimeFilter",
    ]
    payload = {
        "scope": "synthetic_single_batch_only",
        "input_shape": [2, 144, 16],
        "requested_output_shape": [2, 10, 16],
        "no_dataset_loaded": True,
        "no_training_started": True,
        "results": [run_one(name) for name in names],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
