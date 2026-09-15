param(
    [string]$Python = "python",
    [string]$GeneratorCheckpoint,
    [string]$QvinaBinary = "$env:CONDA_PREFIX\Library\bin\qvina2.EXE",
    [string]$ObabelBinary = "$env:CONDA_PREFIX\Library\bin\obabel.EXE",
    [string]$OutputDir = "../outputs/pilot",
    [string]$Config = "../configs/pilot.yaml",
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$SkipSelect
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Project = Split-Path -Parent $Here
$Main = Split-Path -Parent $Project
$DefaultEnv = Join-Path $env:USERPROFILE "miniconda3\envs\diffsbdd"
if ($Python -eq "python") {
    $EnvPython = Join-Path $DefaultEnv "python.exe"
    if (Test-Path $EnvPython) { $Python = $EnvPython }
}
if (-not $env:CONDA_PREFIX) { $env:CONDA_PREFIX = $DefaultEnv }
if (-not (Test-Path $QvinaBinary)) {
    $QvinaBinary = Join-Path $DefaultEnv "Library\bin\qvina2.EXE"
}
if (-not (Test-Path $ObabelBinary)) {
    $ObabelBinary = Join-Path $DefaultEnv "Library\bin\obabel.EXE"
}
$Config = [System.IO.Path]::GetFullPath((Join-Path $Here $Config))
$Out = [System.IO.Path]::GetFullPath((Join-Path $Here $OutputDir))
$Eval = Join-Path $Out "eval"
if (-not $GeneratorCheckpoint) {
    $GeneratorCheckpoint = Join-Path $Main "model\checkpoints\crossdocked_fullatom_cond.ckpt"
}
foreach ($required in @($GeneratorCheckpoint, $QvinaBinary, $ObabelBinary)) {
    if (-not (Test-Path $required)) { throw "Required executable/file missing: $required" }
}
New-Item -ItemType Directory -Force -Path $Out | Out-Null

function Run-Python {
    $env:PYTHONUNBUFFERED = "1"
    & $Python -u @args
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $($args -join ' ')" }
}

if (-not $SkipSelect) {
    Run-Python (Join-Path $Here "01_select_pockets.py") --config $Config --output-dir $Out
}
if (-not $SkipTrain) {
    Run-Python (Join-Path $Here "02_train.py") --config $Config `
        --ft-pockets (Join-Path $Out "ft_pockets.csv") --output-dir $Out `
        --qvina-binary $QvinaBinary --obabel-binary $ObabelBinary
}
$Ckpt = Join-Path $Out "last.ckpt"
if (-not (Test-Path $Ckpt)) { throw "Missing fine-tuned checkpoint: $Ckpt" }
if (-not $SkipEval) {
    Run-Python (Join-Path $Here "03_evaluate.py") --config $Config `
        --test-pockets (Join-Path $Out "test_pockets.csv") `
        --ftdiff-checkpoint $Ckpt --output-dir $Eval `
        --qvina-binary $QvinaBinary --obabel-binary $ObabelBinary
    & $Python (Join-Path $Here "04_write_report.py") --eval-dir $Eval
}
