#!/usr/bin/env python3
"""
板端资源测量：延迟 + 进程 RSS + NPU 平均负载/忙碌时间估计。

仅标准库 + librknnrt.so。在连续推理期间后台线程采样：
  - /proc/self/status 的 VmRSS / VmHWM
  - /sys/kernel/debug/rknpu/load 三核利用率

忙碌时间估计（启发式，非功率计）:
  busy_ms_per_frame ≈ latency_mean_ms * mean(NPU_core_avg_load%)

用法:
  python3 resource_bench.py --rknn student_nuaa.rknn --tag student --iters 100
  python3 resource_bench.py --rknn teacher_nuaa.rknn --tag teacher --iters 100
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import os
import re
import threading
import time
from datetime import datetime

# 复用同目录 ctypes bench 的结构定义
import rknn_ctypes_bench as base

NPU_LOAD_PATH = "/sys/kernel/debug/rknpu/load"


def read_self_rss_kb():
    rss = hwm = None
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1])
    except OSError:
        pass
    return rss, hwm


def parse_npu_loads(text):
    """解析 'Core0: 41%, Core1: 20%, Core2: 20%' -> [41,20,20]"""
    if not text:
        return []
    return [int(x) for x in re.findall(r"Core\d+:\s*(\d+)%", text)]


def read_npu_raw():
    if not os.path.exists(NPU_LOAD_PATH):
        return ""
    try:
        with open(NPU_LOAD_PATH, "r") as f:
            return f.read().strip().replace("\n", " ")
    except OSError:
        return ""


class Sampler(threading.Thread):
    def __init__(self, interval_s=0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()  # 勿命名 _stop，会覆盖 Thread._stop
        self.rss_samples = []
        self.hwm_samples = []
        self.npu_samples = []  # list of [c0,c1,c2]
        self.npu_raw_last = ""

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            rss, hwm = read_self_rss_kb()
            if rss is not None:
                self.rss_samples.append(rss)
            if hwm is not None:
                self.hwm_samples.append(hwm)
            raw = read_npu_raw()
            self.npu_raw_last = raw
            loads = parse_npu_loads(raw)
            if loads:
                self.npu_samples.append(loads)
            time.sleep(self.interval_s)


def mean(xs):
    return sum(xs) / float(len(xs)) if xs else 0.0


def percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def main():
    p = argparse.ArgumentParser(description="RK3588 resource bench (RSS + NPU busy)")
    p.add_argument("--rknn", required=True)
    p.add_argument("--tag", default="model")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--core_mask", type=int, default=7)
    p.add_argument("--sample_hz", type=float, default=20.0, help="采样频率 Hz")
    p.add_argument("--csv", default="resource_bench.csv")
    p.add_argument("--lib", default=base.LIB_PATH)
    args = p.parse_args()

    if not os.path.isfile(args.rknn):
        raise SystemExit("missing rknn: %s" % args.rknn)

    lib = ctypes.CDLL(args.lib)
    base.bind_lib(lib)

    ctx = ctypes.c_uint64(0)
    path = ctypes.c_char_p(os.path.abspath(args.rknn).encode("utf-8"))
    base.check(lib.rknn_init(ctypes.byref(ctx), path, 0, base.RKNN_FLAG_PRIOR_HIGH, None), "rknn_init")

    if hasattr(lib, "rknn_set_core_mask"):
        lib.rknn_set_core_mask(ctx.value, int(args.core_mask))

    version = base.RKNNSdkVersion()
    base.check(
        lib.rknn_query(ctx.value, base.RKNN_QUERY_SDK_VERSION, ctypes.byref(version), ctypes.sizeof(version)),
        "query sdk",
    )

    in_attr = base.RKNNTensorAttr()
    in_attr.index = 0
    base.check(
        lib.rknn_query(ctx.value, base.RKNN_QUERY_INPUT_ATTR, ctypes.byref(in_attr), ctypes.sizeof(in_attr)),
        "query input",
    )
    dims = list(in_attr.dims[: in_attr.n_dims])
    raw, dtype, pass_through, fmt = base.make_uint8_nhwc(dims)
    buf = ctypes.create_string_buffer(raw)
    inputs = (base.RKNNInput * 1)()
    inputs[0].index = 0
    inputs[0].buf = ctypes.cast(buf, ctypes.c_void_p)
    inputs[0].size = len(raw)
    inputs[0].pass_through = pass_through
    inputs[0].type = dtype
    inputs[0].fmt = fmt

    def one_run():
        base.check(lib.rknn_inputs_set(ctx.value, 1, inputs), "inputs_set")
        base.check(lib.rknn_run(ctx.value, None), "run")
        outs = (base.RKNNOutput * 1)()
        outs[0].want_float = 1
        outs[0].is_prealloc = 0
        outs[0].index = 0
        base.check(lib.rknn_outputs_get(ctx.value, 1, outs, None), "outputs_get")
        lib.rknn_outputs_release(ctx.value, 1, outs)

    # idle baseline RSS（初始化后、推理前）
    time.sleep(0.2)
    rss_idle, hwm_idle = read_self_rss_kb()

    for _ in range(args.warmup):
        one_run()

    sampler = Sampler(interval_s=1.0 / max(args.sample_hz, 1.0))
    sampler.start()
    lat = []
    t_wall0 = time.perf_counter()
    for _ in range(args.iters):
        t0 = time.perf_counter()
        one_run()
        lat.append((time.perf_counter() - t0) * 1000.0)
    wall_s = time.perf_counter() - t_wall0
    sampler.stop()
    sampler.join(timeout=2.0)

    rss_end, hwm_end = read_self_rss_kb()
    lat_sorted = sorted(lat)
    lat_mean = mean(lat)

    # NPU samples -> per-core mean
    npu_means = [0.0, 0.0, 0.0]
    npu_avg = 0.0
    if sampler.npu_samples:
        n = len(sampler.npu_samples)
        for sample in sampler.npu_samples:
            for i, v in enumerate(sample[:3]):
                npu_means[i] += v
        npu_means = [x / n for x in npu_means]
        npu_avg = mean(npu_means)

    rss_mean = mean(sampler.rss_samples) if sampler.rss_samples else (rss_end or 0)
    rss_peak = max(sampler.rss_samples) if sampler.rss_samples else (hwm_end or 0)
    hwm_peak = max(sampler.hwm_samples) if sampler.hwm_samples else (hwm_end or 0)

    # 忙碌时间估计：每帧墙钟 × 平均利用率
    busy_ms = lat_mean * (npu_avg / 100.0)
    # 整段忙碌时间（秒）
    busy_total_s = wall_s * (npu_avg / 100.0)

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "rknn": os.path.abspath(args.rknn),
        "rknn_mb": round(os.path.getsize(args.rknn) / 1024.0 / 1024.0, 3),
        "input_dims": str(dims),
        "core_mask": args.core_mask,
        "warmup": args.warmup,
        "iters": args.iters,
        "sample_hz": args.sample_hz,
        "latency_mean_ms": round(lat_mean, 3),
        "latency_std_ms": round((mean([(x - lat_mean) ** 2 for x in lat]) ** 0.5), 3),
        "latency_p50_ms": round(percentile(lat_sorted, 50), 3),
        "latency_p99_ms": round(percentile(lat_sorted, 99), 3),
        "fps": round(1000.0 / lat_mean, 2) if lat_mean > 0 else 0,
        "wall_s": round(wall_s, 3),
        "rss_idle_kB": rss_idle,
        "rss_mean_kB": int(round(rss_mean)),
        "rss_peak_kB": int(round(rss_peak)),
        "rss_end_kB": rss_end,
        "VmHWM_peak_kB": int(round(hwm_peak)),
        "rss_delta_vs_idle_kB": (int(round(rss_mean)) - rss_idle) if rss_idle is not None else None,
        "npu_core0_mean_pct": round(npu_means[0], 2),
        "npu_core1_mean_pct": round(npu_means[1], 2),
        "npu_core2_mean_pct": round(npu_means[2], 2),
        "npu_avg_pct": round(npu_avg, 2),
        "npu_samples": len(sampler.npu_samples),
        "est_busy_ms_per_frame": round(busy_ms, 3),
        "est_busy_total_s": round(busy_total_s, 3),
        "api": base.c_string(version.api_version),
        "drv": base.c_string(version.drv_version),
        "note": "busy≈latency*avg_npu%; energy needs power meter; RSS=process VmRSS",
    }

    print("======== Resource Bench [%s] ========" % args.tag)
    for k, v in summary.items():
        print("  %s: %s" % (k, v))
    print("====================================")

    write_header = not os.path.isfile(args.csv)
    with open(args.csv, "a") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print("[save]", os.path.abspath(args.csv))

    # 采样明细
    detail = os.path.splitext(args.csv)[0] + "_%s_npu_samples.txt" % args.tag
    with open(detail, "w") as f:
        f.write("c0,c1,c2\n")
        for s in sampler.npu_samples:
            while len(s) < 3:
                s.append(0)
            f.write("%d,%d,%d\n" % (s[0], s[1], s[2]))
    print("[save]", os.path.abspath(detail))

    lib.rknn_destroy(ctx.value)


if __name__ == "__main__":
    main()
