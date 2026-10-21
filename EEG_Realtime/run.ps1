param([int]$Port = 8765, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$candidates = @(
    (Join-Path $PSScriptRoot '.venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'),
    'D:\Anaconda\python.exe'
)
$pinnedRuntimeFile = Join-Path $PSScriptRoot '.packages\runtime.json'
if (Test-Path -LiteralPath $pinnedRuntimeFile -PathType Leaf) {
    $pinnedRuntime = Get-Content -LiteralPath $pinnedRuntimeFile -Raw | ConvertFrom-Json
    if (-not $pinnedRuntime.executable -or -not (Test-Path -LiteralPath $pinnedRuntime.executable -PathType Leaf)) {
        throw 'The configured EEG Python runtime is missing. Restore it or reconfigure dependencies and the EEG firewall together.'
    }
    # Keep manual launches on the same executable as the managed service and
    # its program-scoped firewall rule; never silently switch to Anaconda.
    $candidates = @([IO.Path]::GetFullPath($pinnedRuntime.executable))
}
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not (Test-Path -LiteralPath $pinnedRuntimeFile -PathType Leaf) -and $pythonCommand -and $pythonCommand.Source -notlike '*WindowsApps*') {
    $candidates += $pythonCommand.Source
}
$runtime = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        # Windows PowerShell 5.1 turns native stderr into terminating errors when
        # ErrorActionPreference is Stop. Keep a failed probe local and try the
        # next interpreter instead of blocking an otherwise usable fallback.
        $probePreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = 'Continue'
            & $candidate -B -c 'import eeg_realtime; import sys, numpy, scipy, sklearn, joblib; assert sys.version_info >= (3, 10)' 2>$null
            $probeExitCode = $LASTEXITCODE
        } catch {
            $probeExitCode = 1
        } finally {
            $ErrorActionPreference = $probePreference
        }
        if ($probeExitCode -eq 0) { $runtime = $candidate; break }
    }
}
if (-not $runtime) {
    Write-Host 'Python 3.10+, NumPy, SciPy, scikit-learn and Joblib are required. See README.md for setup.' -ForegroundColor Red
    exit 1
}
Write-Host "Python: $runtime"
$serverArgs = @('-B', '-m', 'eeg_realtime.server', '--port', "$Port")
if (-not $NoBrowser) { $serverArgs += '--open' }
& $runtime @serverArgs
exit $LASTEXITCODE
