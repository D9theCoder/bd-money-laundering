# Run a command in the background and log its output:
#   powershell -File bg.ps1 -Name <name> -Cmd "<command line>"
param([string]$Name, [string]$Cmd)
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\bg_$Name.log"
"START $(Get-Date -Format o) :: $Cmd" | Out-File -Encoding utf8 $log
$c = "($Cmd >> $log 2>&1) && (echo EXIT 0 >> $log) || (echo EXIT 1 >> $log)"
Start-Process -FilePath cmd.exe -ArgumentList '/c', $c -WindowStyle Hidden -WorkingDirectory $PSScriptRoot
