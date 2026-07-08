import os
import sys
import torch
import typer

from occwl.io import load_step

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brepdiff.config import Config, load_config
from brepdiff.models.brepdiff import BrepDiff

app = typer.Typer(pretty_exceptions_enable=False)


def load_model(ckpt_path: str, device: str = "cuda") -> BrepDiff:
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(ckpt_path)), "config.yaml"
    )
    config = load_config(config_path, "")
    config.log_dir = os.path.dirname(os.path.dirname(ckpt_path))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            state_dict[k[len("model."):]] = v

    model = BrepDiff(config, acc_logger=None)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@app.command()
def main(
    ckpt_path: str = typer.Option(..., help="체크포인트 경로"),
    condition_path: str = typer.Option(..., help="조건 STEP 파일 경로"),
    n_samples: int = typer.Option(10, help="생성할 샘플 수"),
    cfg_scale: float = typer.Option(1.0, help="Classifier-free guidance scale"),
    output_dir: str = typer.Option("results/cond_only", help="결과 저장 디렉토리"),
    num_faces: int = typer.Option(10, help="조건 STEP에서 사용할 최대 face 수 (기본값: 모델의 seq_len)"),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading model from {ckpt_path} ...")
    model = load_model(ckpt_path, device=device)

    # condition 이름으로 하위 디렉토리 생성
    solids = load_step(condition_path)
    if len(solids) == 0:
        raise ValueError(f"No solids found in condition STEP: {condition_path}")

    solid = solids[0]
    total_faces = len(list(solid.faces()))
    if total_faces <= 0:
        raise ValueError(f"Condition solid has no faces: {condition_path}")

    num_faces = min(num_faces, model.seq_len)
    print(f"Condition has {num_faces} faces. Sampling with this number of faces.")
    print(f"Sampling {n_samples} samples from condition: {condition_path}")

    result = model.sample_cond_only(
        condition_path=condition_path,
        n_samples=n_samples,
        num_faces=num_faces,
        cfg_scale=cfg_scale,
        save_dir=output_dir,
    )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for i in range(n_samples):
            npz_path = os.path.join(output_dir, f"{str(i).zfill(5)}.npz")
            result.uvgrids.export_npz(file_path=npz_path, vis_idx=i)
            print(f"Saved: {npz_path}")

    print(f"Done. Results saved to {output_dir}/")

if __name__ == "__main__":
    app()
