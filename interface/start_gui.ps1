$ErrorActionPreference='Stop'
$taskGuiRoot=Split-Path -Parent $MyInvocation.MyCommand.Path
$taskPython=Join-Path $taskGuiRoot '.venv\Scripts\python.exe'
if(-not (Test-Path -LiteralPath $taskPython)){throw 'Prediction runtime is missing. Run install_runtime.ps1 first (see interface/README.md).'}
$taskReady=$false
try{$taskHealth=Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 2;$taskReady=($taskHealth.app -eq 'Muscatine biogas predictor')}catch{}
if(-not $taskReady){
    Start-Process -FilePath $taskPython -ArgumentList @('"'+(Join-Path $taskGuiRoot 'server.py')+'"') -WorkingDirectory $taskGuiRoot -WindowStyle Hidden
    for($taskTry=0;$taskTry -lt 60;$taskTry++){Start-Sleep -Milliseconds 500;try{$taskHealth=Invoke-RestMethod 'http://127.0.0.1:8765/health' -TimeoutSec 1;if($taskHealth.app -eq 'Muscatine biogas predictor'){$taskReady=$true;break}}catch{}}
}
if(-not $taskReady){throw 'The prediction server did not start. Check port 8765 and run server.py to view the error.'}
Start-Process 'http://127.0.0.1:8765/'
