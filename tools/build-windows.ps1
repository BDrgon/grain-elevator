param(
    [string]$Model = 'clap-fp32',
    [string]$Python = 'python',
    [string]$VcRedist,
    [switch]$Offline,
    [switch]$AllowSystemCrt
)
$ErrorActionPreference = 'Stop'
$buildArgs = @("$PSScriptRoot/build.py", '--platform', 'windows-x64', '--model', $Model)
if ($VcRedist) { $buildArgs += @('--vc-redist', $VcRedist) }
if ($Offline) { $buildArgs += '--offline' }
if ($AllowSystemCrt) { $buildArgs += '--allow-system-crt' }
& $Python @buildArgs
exit $LASTEXITCODE
