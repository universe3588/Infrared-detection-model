# Ex04：三有权重基线 × 四数据集评测
# 用法：
#   .\scripts\run_ex04_baselines.ps1
#   .\scripts\run_ex04_baselines.ps1 -Limit 8
#   .\scripts\run_ex04_baselines.ps1 -Models "IRPruneDet,LW" -Datasets "NUAA-SIRST"

param(
    [string]$Models = "IRPruneDet,LW,RepISD",
    [string]$Datasets = "NUAA-SIRST,NUDT-SIRST,IRSTD-1k,SIRST-V2",
    [string]$Device = "",
    [int]$Limit = 0
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$argsList = @(
    "scripts/eval_ex04_baselines.py",
    "--models", $Models,
    "--datasets", $Datasets
)
if ($Limit -gt 0) { $argsList += @("--limit", "$Limit") }
if ($Device -ne "") { $argsList += @("--device", $Device) }

Write-Host "python $($argsList -join ' ')"
python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
