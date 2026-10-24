from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import mlflow
import yaml
from mlflow import MlflowClient

from mlops.train import MODEL_NAME


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, **kw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--tol", type=float, default=0.005)
    a = ap.parse_args()

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, str(a.model_version))
    data_v, commit = mv.tags["data_version"], mv.tags["git_commit"]
    original = client.get_run(mv.run_id).data.metrics["val_pr_auc"]
    print(f"model v{a.model_version}: data_version={data_v} git_commit={commit} original val_pr_auc={original:.4f}")

    main_repo = Path.cwd()
    wt = Path(tempfile.mkdtemp(prefix="repro-")) / "repo"
    run(["git", "worktree", "add", "--detach", str(wt), commit])
    try:
        shutil.copy(main_repo / ".dvc" / "config.local", wt / ".dvc" / "config.local")
        run(["dvc", "config", "--local", "cache.dir", str((main_repo / ".dvc" / "cache").resolve())], cwd=wt)
        run(["dvc", "pull"], cwd=wt)

        got = yaml.safe_load((wt / "data" / "raw.dvc").read_text())["outs"][0]["md5"]
        assert got == data_v, f"data version mismatch: {got} != {data_v}"

        env = {**os.environ, "PYTHONPATH": str(wt / "src"), "RAW_DIR": "data/raw", "INTERIM_DIR": "data/interim"}
        run([sys.executable, "-m", "mlops.run_batch", "--from", "2016-03", "--to", "2017-05"], cwd=wt, env=env)
        run([sys.executable, "-m", "mlops.train",
             "--extra-tag", f"reproduces_model_version={a.model_version}"], cwd=wt, env=env)

        import json
        new = json.loads((wt / "reports" / "last_run.json").read_text())["pr_auc"]
    finally:
        run(["git", "worktree", "remove", "--force", str(wt)])

    diff = abs(new - original)
    verdict = "PASS" if diff <= a.tol else "FAIL"
    print(f"\noriginal={original:.4f} reproduced={new:.4f} |diff|={diff:.4f} tol={a.tol} -> {verdict}")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()