$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDirectory
try {
    & "$ProjectDirectory\.venv\Scripts\python.exe" -c "import torch,sys; from repurpose.academic.cli import main; sys.argv=['academic','resume']; main()"
    if ($LASTEXITCODE -ne 0) { throw 'Academic foundation validation failed.' }
} finally { Pop-Location }
