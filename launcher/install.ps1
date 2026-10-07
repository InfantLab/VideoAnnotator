# Installs videoannotator-start for Windows, and a "Start VideoAnnotator"
# shortcut on the desktop:
#
#   powershell -ExecutionPolicy Bypass -c "irm https://github.com/InfantLab/VideoAnnotator/releases/latest/download/install.ps1 | iex"
#
# Copies the launcher into %LOCALAPPDATA%\VideoAnnotator and adds that folder
# to your PATH; nothing else is installed (spec 024, research R16).

param([switch]$NoShortcut, [switch]$Quiet)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$github = 'InfantLab/VideoAnnotator'
# Replaced with the release version by CI. `videoannotator-start update` sets
# VA_RELEASE to the release it is updating to.
$release = '@VERSION@'
if ($env:VA_RELEASE) { $release = $env:VA_RELEASE }
$dir = Join-Path $env:LOCALAPPDATA 'VideoAnnotator'
$script:Silent = [bool]$Quiet

function Write-Step {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '', Justification = 'An installer talking to its user')]
    param([string]$Text)
    if (-not $script:Silent) { Write-Host $Text }
}

if ($release -eq '@VERSION@') {
    $base = "https://github.com/$github/releases/latest/download"
} elseif ($release.StartsWith('v')) {
    $base = "https://github.com/$github/releases/download/$release"
} else {
    $base = "https://github.com/$github/releases/download/v$release"
}

[void][IO.Directory]::CreateDirectory($dir)
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
foreach ($file in @('videoannotator-start.ps1', 'videoannotator-start.cmd')) {
    try {
        Invoke-WebRequest -Uri "$base/$file" -OutFile (Join-Path $dir $file) -UseBasicParsing
    } catch {
        # Not exit: under `irm | iex` that would close the researcher's window.
        throw "Couldn't download VideoAnnotator. Check your internet connection and run this again."
    }
}
Write-Step "Installed videoannotator-start in $dir."

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath) { $userPath = '' }
if (($userPath.Split(';') | Where-Object { $_ -ieq $dir }).Count -eq 0) {
    [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ";$dir").TrimStart(';'), 'User')
    Write-Step 'Added it to your PATH: open a new terminal to use videoannotator-start.'
}

if (-not $NoShortcut) {
    $shell = New-Object -ComObject WScript.Shell
    $link = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Desktop')) 'Start VideoAnnotator.lnk'))
    $link.TargetPath = Join-Path $dir 'videoannotator-start.cmd'
    $link.WorkingDirectory = $env:USERPROFILE
    $link.Description = 'Start VideoAnnotator and open it in your browser'
    $link.Save()
    Write-Step 'Added "Start VideoAnnotator" to your desktop.'
}

Write-Step ''
Write-Step 'To start VideoAnnotator, double-click "Start VideoAnnotator", or run: videoannotator-start'
