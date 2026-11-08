#!/usr/bin/env bash
# 兼容旧入口：转调更稳方案（NUAA A+B + NUDT 仅 A）
set -euo pipefail
cd "$(dirname "$0")"
sed -i 's/\r$//' run_dna_ablation_stable.sh 2>/dev/null || true
exec bash run_dna_ablation_stable.sh "$@"
