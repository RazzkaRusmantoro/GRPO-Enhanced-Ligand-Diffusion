$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Yaml = Join-Path $Root "environment_windows.yaml"

Write-Host "=== Create diffsbdd env (Windows) ===" -ForegroundColor Cyan
Write-Host "yaml: $Yaml"

conda env create -f $Yaml -n diffsbdd
if ($LASTEXITCODE -ne 0) { throw "conda env create failed" }

Write-Host "=== Install torch-scatter wheel (pip) ===" -ForegroundColor Cyan
conda run -n diffsbdd pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
if ($LASTEXITCODE -ne 0) { throw "torch-scatter install failed" }

Write-Host "=== Quick import check ===" -ForegroundColor Cyan
conda run -n diffsbdd python -c "import torch; import torch_scatter; import pytorch_lightning; import rdkit; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('OK')"

Write-Host ""
Write-Host "Next: conda activate diffsbdd"
Write-Host "Then: python 01_setup_check.py"
