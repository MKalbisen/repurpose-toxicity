param([ValidateSet('prepare','pilot','freeze','train-toxicity','train-kg','evaluate-toxicity','evaluate-kg','export')][string]$Command = 'pilot')
$ErrorActionPreference = 'Stop'
$ProjectDirectory = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectDirectory
try {
    & "$ProjectDirectory\.venv\Scripts\python.exe" -m repurpose.academic.experiment $Command
    if ($LASTEXITCODE -ne 0) { throw "Academic command failed: $Command" }
} finally { Pop-Location }
