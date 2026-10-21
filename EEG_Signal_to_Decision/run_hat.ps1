param([Parameter(ValueFromRemainingArguments=$true)][string[]]$PipelineArgs)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$realtimeRoot = Join-Path (Split-Path $PSScriptRoot -Parent) 'EEG_Realtime'
$candidates = @(
    (Join-Path $realtimeRoot '.venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
    'D:\Anaconda\python.exe'
)
$pathPython = Get-Command python -ErrorAction SilentlyContinue
if ($pathPython -and $pathPython.Source -notlike '*WindowsApps*') { $candidates += $pathPython.Source }
$runtime = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        $oldPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $candidate -B hat_cli.py --check-runtime 2>$null
            if ($LASTEXITCODE -eq 0) { $runtime = $candidate }
        } finally { $ErrorActionPreference = $oldPreference }
        if ($runtime) { break }
    }
}
if (-not $runtime) { throw 'Install EEG_Realtime/requirements.txt in a Python 3.10+ environment first.' }
& $runtime -B -X utf8 hat_cli.py @PipelineArgs
exit $LASTEXITCODE
