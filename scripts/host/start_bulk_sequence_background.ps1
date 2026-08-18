param(
    [Parameter(Mandatory = $true)][string]$PythonExecutable,
    [Parameter(Mandatory = $true)][string]$Sequence,
    [Parameter(Mandatory = $true)][string]$BlenderExecutable,
    [Parameter(Mandatory = $true)][string]$StdoutLog,
    [Parameter(Mandatory = $true)][string]$StderrLog
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$stdoutPath = Join-Path $repoRoot $StdoutLog
$stderrPath = Join-Path $repoRoot $StderrLog
New-Item -ItemType Directory -Force -Path (Split-Path $stdoutPath), (Split-Path $stderrPath) | Out-Null
$arguments = @(
    "scripts/host/run_bulk_sequence.py",
    "--sequence", $Sequence,
    "--blender-executable", ('"' + $BlenderExecutable + '"')
)
$process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList $arguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru
$sequenceName = [System.IO.Path]::GetFileNameWithoutExtension($Sequence)
$pidPath = Join-Path (Split-Path $stdoutPath) ($sequenceName + ".pid")
Set-Content -LiteralPath $pidPath -Value $process.Id -Encoding ascii
[PSCustomObject]@{
    process_id = $process.Id
    pid_file = $pidPath
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
} | ConvertTo-Json
