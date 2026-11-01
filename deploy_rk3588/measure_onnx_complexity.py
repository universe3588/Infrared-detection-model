"""
本地无 torch 时：从 ONNX 统计参数量，并估算 Conv MACs/FLOPs。

  python measure_onnx_complexity.py \
    --teacher outputs/teacher/teacher_nuaa.onnx \
    --student outputs/student_nuaa.onnx \
    --out outputs/complexity_nuaa_onnx.csv
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import onnx
from onnx import numpy_helper


def initializer_params(model):
    total = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        total += int(arr.size)
    return total


def _get_shape(value_info_map, name):
    vi = value_info_map.get(name)
    if vi is None or not vi.type.tensor_type.HasField("shape"):
        return None
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            return None
    return dims


def conv_macs(weight_shape, out_spatial_hw, groups=1):
    # weight: [Cout, Cin/groups, kH, kW]
    if len(weight_shape) != 4:
        return 0
    cout, cin_g, kh, kw = weight_shape
    h, w = out_spatial_hw
    return int(h * w * cout * cin_g * kh * kw * groups)


def estimate_macs(model):
    # 需要 shape 推理
    try:
        from onnx import shape_inference

        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print("[warn] shape_inference failed:", e)

    init_map = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    vi_map = {}
    for vi in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        vi_map[vi.name] = vi

    macs = 0
    counted_convs = 0
    skipped = 0
    for node in model.graph.node:
        if node.op_type not in ("Conv", "ConvInteger"):
            continue
        w_name = node.input[1] if len(node.input) > 1 else None
        if w_name not in init_map:
            skipped += 1
            continue
        w = init_map[w_name]
        out_name = node.output[0]
        out_shape = _get_shape(vi_map, out_name)
        if out_shape is None or len(out_shape) < 4:
            skipped += 1
            continue
        # NCHW
        h, w_sp = out_shape[-2], out_shape[-1]
        groups = 1
        for a in node.attribute:
            if a.name == "group":
                groups = int(a.i)
        macs += conv_macs(w.shape, (h, w_sp), groups=groups)
        counted_convs += 1

    return macs, counted_convs, skipped


def measure(path, name):
    model = onnx.load(path)
    params = initializer_params(model)
    macs, n_conv, n_skip = estimate_macs(model)
    flops = macs * 2
    return {
        "name": name,
        "path": os.path.abspath(path),
        "file_mb": round(os.path.getsize(path) / 1024 / 1024, 3),
        "params": params,
        "params_M": round(params / 1e6, 4),
        "macs": macs,
        "macs_G": round(macs / 1e9, 4),
        "flops": flops,
        "flops_G": round(flops / 1e9, 4),
        "conv_counted": n_conv,
        "conv_skipped": n_skip,
        "method": "onnx_initializer+conv_macs",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--out", default="outputs/complexity_nuaa_onnx.csv")
    args = p.parse_args()

    rows = [
        measure(args.teacher, "teacher"),
        measure(args.student, "student"),
    ]
    t, s = rows[0], rows[1]
    rows.append(
        {
            "name": "delta_student_vs_teacher",
            "path": "",
            "file_mb": round(s["file_mb"] - t["file_mb"], 3),
            "params": s["params"] - t["params"],
            "params_M": round(s["params_M"] - t["params_M"], 4),
            "macs": s["macs"] - t["macs"],
            "macs_G": round(s["macs_G"] - t["macs_G"], 4),
            "flops": s["flops"] - t["flops"],
            "flops_G": round(s["flops_G"] - t["flops_G"], 4),
            "conv_counted": "",
            "conv_skipped": "",
            "method": "params%%=%.2f; flops%%=%.2f"
            % (100 * (s["params"] / t["params"] - 1), 100 * (s["flops"] / max(t["flops"], 1) - 1)),
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print("======== ONNX Complexity ========")
    for r in rows:
        print(
            f"{r['name']}: params={r['params_M']}M  MACs={r['macs_G']}G  "
            f"FLOPs={r['flops_G']}G  file={r['file_mb']}MB"
        )
    print(f"[save] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
