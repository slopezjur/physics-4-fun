param([string]$BasePython = "python")
$ErrorActionPreference = "Stop"
$EnvPath = Join-Path $PSScriptRoot ".venv"
$PythonPath = Join-Path $EnvPath "Scripts/python.exe"
$UvPath = Join-Path $EnvPath "Scripts/uv.exe"

& $BasePython -c "import sys; assert sys.version_info[:2] == (3, 12), 'Use Python 3.12 for this validated environment'"
if ($LASTEXITCODE -ne 0) { throw "Unsupported Python version" }
if (-not (Test-Path -LiteralPath $PythonPath)) {
    & $BasePython -m venv $EnvPath
    if ($LASTEXITCODE -ne 0) { throw "Failed to create isolated environment" }
}
& $PythonPath -m pip install "uv==0.12.17"
if ($LASTEXITCODE -ne 0) { throw "Failed to install uv" }
& $UvPath --system-certs pip install --python $PythonPath --index https://download.pytorch.org/whl/cu128 "torch==2.11.0+cu128"
if ($LASTEXITCODE -ne 0) { throw "Failed to install pinned CUDA Torch" }
& $UvPath --system-certs pip install --python $PythonPath -r (Join-Path $PSScriptRoot "requirements.lock.txt")
if ($LASTEXITCODE -ne 0) { throw "Failed to install pinned requirements" }
& $UvPath pip check --python $PythonPath
if ($LASTEXITCODE -ne 0) { throw "Dependency validation failed" }
Write-Output "Ready: $PythonPath"
