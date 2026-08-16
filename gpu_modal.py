"""Run the gather measurement on an NVIDIA GPU through Modal, since the class project has no H100 quota.

The class GCP project has A100 and L4 quota but zero H100 in every region checked, and the identity
running this study cannot call `compute.instances.create` anyway. Modal rents an H100 by the second,
which is the cheapest way to put a real Hopper number next to the TPU ones. At roughly $4/hour a
five-minute run costs about thirty cents.

    modal run gpu_modal.py                       # H100, the span/order/allocation grid
    modal run gpu_modal.py --sweep                # H100, the allocation ladder
    modal run gpu_modal.py --gpu A100-80GB        # for the GCP-comparable part
"""
from __future__ import annotations

import json
import pathlib

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch")
    .add_local_file(HERE / "gpu_gather.py", "/root/gpu_gather.py", copy=True)
)

app = modal.App("tpu-study-gather")


@app.function(image=image, gpu="H100", timeout=1800)
def bench(sweep: bool = False) -> dict:
    import subprocess
    import sys

    cmd = [sys.executable, "/root/gpu_gather.py", "--out", "/tmp/out.json"]
    if sweep:
        cmd.append("--sweep")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-4000:])
    try:
        return {"stdout": proc.stdout, "json": json.loads(pathlib.Path("/tmp/out.json").read_text())}
    except Exception:
        return {"stdout": proc.stdout, "stderr": proc.stderr[-4000:], "json": None}


@app.local_entrypoint()
def main(sweep: bool = False, out: str = "gpu_h100.json"):
    result = bench.remote(sweep=sweep)
    print(result["stdout"])
    if result.get("json"):
        pathlib.Path(out).write_text(json.dumps(result["json"], indent=2))
        print(f"wrote {out}")
    else:
        print("no JSON came back:", result.get("stderr", "")[-2000:])
