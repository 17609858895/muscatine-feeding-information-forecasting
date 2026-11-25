$ErrorActionPreference='Stop'
$taskGuiRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskLocalEnvironment=Join-Path $taskGuiRoot '.venv'
$taskPython=Join-Path $taskLocalEnvironment 'Scripts\python.exe'
if(-not (Test-Path -LiteralPath $taskPython)){
    & py -3.12 -m venv $taskLocalEnvironment
    if($LASTEXITCODE -ne 0){throw 'Python 3.12 is required. Install Python from python.org and rerun this script.'}
}
& $taskPython -m pip install -r (Join-Path $taskGuiRoot 'requirements.txt')
if($LASTEXITCODE -ne 0){throw 'Dependency installation failed. Check the pip output.'}
Write-Host 'Runtime ready. Run start_gui.ps1 to open the local predictor.'
