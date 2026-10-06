# videoannotator-start for Windows: start VideoAnnotator in Docker Desktop or
# Podman Desktop, sharing the folders a researcher chooses, read-only (spec 024).
#
# The same behaviour as the sh script beside it, from the same test cases
# (tests/launcher/cases.json); launcher/README.md explains the layout. Windows
# PowerShell 5.1 and later. ASCII only: 5.1 reads a file without a BOM as ANSI.
#
#   videoannotator-start [share|unshare|list|stop|update|logs] [options]

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Replaced with the release version by CI. Still the placeholder in a source
# checkout, which then runs :latest and says it is a development launcher.
$script:VaVersion = '@VERSION@'
$script:VaImageRepo = 'ghcr.io/infantlab/videoannotator'
$script:VaName = 'videoannotator'
$script:VaInternalPort = 18011
$script:VaDefaultPort = 18011
$script:VaGithub = 'InfantLab/VideoAnnotator'
$script:VaGuide = "https://github.com/$($script:VaGithub)/blob/master/docs/installation/troubleshooting.md"
$script:VaVolumes = @('models', 'database', 'storage', 'cache')

# ---------------------------------------------------------------------------
# Output and questions
# ---------------------------------------------------------------------------

function Write-VaSay {
    [Diagnostics.CodeAnalysis.SuppressMessageAttribute('PSAvoidUsingWriteHost', '', Justification = 'A console program talking to its user')]
    param([string]$Text = '', [switch]$NoNewline)
    if ($NoNewline) { Write-Host $Text -NoNewline } else { Write-Host $Text }
}

function Read-VaAnswer {
    param([string]$Question)
    Write-VaSay "$Question " -NoNewline
    try { $answer = [Console]::ReadLine() } catch { $answer = $null }
    if ($null -eq $answer) { return '' }
    return $answer
}

# 'y' or 'n' default; --yes answers yes.
function Test-VaYes {
    param([string]$Question, [string]$Default)
    if ($script:Yes) { return $true }
    $answer = Read-VaAnswer $Question
    if ($answer -match '^[Yy]') { return $true }
    if ($answer -match '^[Nn]') { return $false }
    return ($Default -eq 'y')
}

# ---------------------------------------------------------------------------
# Settings (data-model.md, research R13)
# ---------------------------------------------------------------------------

function Get-VaSettingsPath {
    param([string]$AppData = $env:APPDATA)
    return ($AppData.TrimEnd('\') + '\VideoAnnotator\start.conf')
}

function Get-VaRequestsDir {
    param([string]$AppData = $env:APPDATA)
    return ($AppData.TrimEnd('\') + '\VideoAnnotator\requests')
}

function Read-VaConfig {
    param([string]$Path = (Get-VaSettingsPath))
    $config = @{ engine = ''; image = ''; results = ''; key = ''; port = ''; shares = @() }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $config }
    foreach ($line in [IO.File]::ReadAllLines($Path)) {
        $at = $line.IndexOf('=')
        if ($at -lt 1) { continue }
        $name = $line.Substring(0, $at)
        $value = $line.Substring($at + 1)
        if ($name -eq 'share') { $config.shares += $value }
        elseif ($config.ContainsKey($name)) { $config[$name] = $value }
    }
    return $config
}

function Write-VaConfig {
    param([hashtable]$Config, [string]$Path = (Get-VaSettingsPath))
    $dir = Split-Path -Parent $Path
    [void][IO.Directory]::CreateDirectory($dir)
    [void][IO.Directory]::CreateDirectory((Join-Path $dir 'requests'))
    $lines = @()
    foreach ($name in @('engine', 'image', 'results', 'port', 'key')) {
        if ($Config[$name]) { $lines += "$name=$($Config[$name])" }
    }
    foreach ($share in $Config.shares) { $lines += "share=$share" }
    # Whole, then moved: never half a file. The folder is the user's own
    # (%APPDATA%), which keeps the key private as on Linux and macOS.
    $temp = "$Path.tmp"
    [IO.File]::WriteAllLines($temp, [string[]]$lines)
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
    Move-Item -LiteralPath $temp -Destination $Path
}

# ---------------------------------------------------------------------------
# Folders (research R3, R4)
# ---------------------------------------------------------------------------

# Absolute, `~` expanded, backslashes, no trailing separator except a drive root.
function ConvertTo-VaNormalPath {
    param([string]$Path, [string]$HomeDir = $env:USERPROFILE)
    $p = $Path.Trim().Replace('/', '\')
    if ($p -eq '~') { $p = $HomeDir }
    elseif ($p.StartsWith('~\')) { $p = $HomeDir.TrimEnd('\') + $p.Substring(1) }
    elseif ($p -notmatch '^[A-Za-z]:\\' -and -not $p.StartsWith('\\')) {
        $p = (Get-Location).ProviderPath.TrimEnd('\') + '\' + $p
    }
    $unc = $p.StartsWith('\\')
    $p = ($p -replace '\\\.(?=\\|$)', '') -replace '\\{2,}', '\'
    if ($unc) { $p = '\' + $p }
    if ($p -match '^[A-Za-z]:\\?$') { return $p.Substring(0, 2) + '\' }
    return $p.TrimEnd('\')
}

# Whether Child is Parent or inside it; Windows paths compare without case.
function Test-VaInside {
    param([string]$Child, [string]$Parent)
    $c = $Child.TrimEnd('\').ToLowerInvariant()
    $p = $Parent.TrimEnd('\').ToLowerInvariant()
    if ($c -eq $p) { return $true }
    return $c.StartsWith($p + '\')
}

# Where a Windows folder is mounted: C:\Users\ada -> /c/Users/ada. A path
# holding a separator the server's settings use (; = ,) or a network path goes
# to /host/<N>. N is the share's number, or "results".
function ConvertTo-VaContainerPath {
    param([string]$Path, [string]$N = '1')
    $p = $Path.TrimEnd('\')
    if ($p -notmatch '^[A-Za-z]:' -or $p.Substring(2) -match '[:;=,]') { return "/host/$N" }
    $drive = $p.Substring(0, 1).ToLowerInvariant()
    $rest = $p.Substring(2).Replace('\', '/')
    return "/$drive$rest"
}

function Get-VaBroadKind {
    param([string]$Path, [string]$HomeDir = $env:USERPROFILE)
    $p = $Path.TrimEnd('\')
    if ($p -ieq $HomeDir.TrimEnd('\')) { return 'home' }
    if ($p -match '^[A-Za-z]:$') { return 'everything' }
    if ($p -match '^[A-Za-z]:\\Users$') { return 'users' }
    if ($p -match '^[A-Za-z]:\\(Windows|Program Files|Program Files \(x86\)|ProgramData)$') { return 'system' }
    return ''
}

function Get-VaBroadMessage {
    param([string]$Kind)
    switch ($Kind) {
        'home' { return 'This shares everything in your home folder, including documents unrelated to your research.' }
        'everything' { return "This shares everything on this drive, including other people's files and system files." }
        'drives' { return 'This shares every drive connected to this computer, including ones unrelated to your research.' }
        'users' { return "This shares every user's home folder on this computer, not only yours." }
        'system' { return 'This shares system files, which hold no videos and may hold private settings.' }
    }
    return ''
}

# What sharing Path would mean: ok | broad <kind> | refused | duplicate, then
# flags replaces, nested_results.
function Get-VaClassification {
    param([string]$Path, [string[]]$Shares = @(), [string]$Results = '', [string]$HomeDir = $env:USERPROFILE)
    if ($Results -and (Test-VaInside $Path $Results)) { return 'refused' }
    $replaces = ''
    foreach ($share in $Shares) {
        if (-not $share) { continue }
        if (Test-VaInside $Path $share) { return 'duplicate' }
        if (Test-VaInside $share $Path) { $replaces = ' replaces' }
    }
    $kind = Get-VaBroadKind -Path $Path -HomeDir $HomeDir
    $out = 'ok'
    if ($kind) { $out = "broad $kind" }
    $out += $replaces
    if ($Results -and (Test-VaInside $Results $Path)) { $out += ' nested_results' }
    return $out
}

# Adds Path to the shares after asking, as the classification decides.
# 0 added, 1 not added (said why), 2 cancelled.
function Add-VaShare {
    param([string]$Path)
    $folder = ConvertTo-VaNormalPath $Path
    if (-not (Test-Path -LiteralPath $folder -PathType Container)) {
        Write-VaSay "Couldn't find the folder $folder."
        return 1
    }
    $class = Get-VaClassification -Path $folder -Shares $script:Config.shares -Results $script:Config.results
    if ($class -eq 'refused') {
        Write-VaSay 'That folder is inside your results folder, which VideoAnnotator can already read.'
        return 1
    }
    if ($class -eq 'duplicate') {
        Write-VaSay "$folder is already shared."
        return 1
    }
    if ($class.StartsWith('broad')) {
        Write-VaSay (Get-VaBroadMessage ($class.Split(' ')[1]))
        if (-not (Test-VaYes 'Share it anyway? [y/N]' 'n')) { return 2 }
    }
    if ($class -match 'replaces') {
        $script:Config.shares = @($script:Config.shares | Where-Object { -not (Test-VaInside $_ $folder) })
    }
    $script:Config.shares = @($script:Config.shares) + $folder
    return 0
}

# ---------------------------------------------------------------------------
# The engine (research R2)
# ---------------------------------------------------------------------------

function Get-VaEngineLabel {
    param([string]$Engine)
    if ($Engine -eq 'podman') { return 'Podman' }
    return 'Docker'
}

# Arguments as one command line, quoted the way Windows programs read them.
function ConvertTo-VaArgString {
    param([string[]]$Arguments)
    $quoted = foreach ($a in $Arguments) {
        if ($a -ne '' -and $a -notmatch '[\s"]') { $a; continue }
        $s = $a -replace '(\\*)"', '$1$1\"'
        $s = $s -replace '(\\+)$', '$1$1'
        '"' + $s + '"'
    }
    return ($quoted -join ' ')
}

# Runs the engine; returns @{ ExitCode; Out; Err }. Not PowerShell's own native
# call: Windows PowerShell 5.1 mangles arguments holding quotes.
function Invoke-VaEngine {
    param([string[]]$Arguments, [string]$Engine = $script:Engine)
    $info = New-Object Diagnostics.ProcessStartInfo
    $info.FileName = $Engine
    $info.Arguments = ConvertTo-VaArgString $Arguments
    $info.UseShellExecute = $false
    $info.RedirectStandardOutput = $true
    $info.RedirectStandardError = $true
    $info.CreateNoWindow = $true
    try {
        $process = [Diagnostics.Process]::Start($info)
    } catch {
        return @{ ExitCode = 127; Out = ''; Err = $_.Exception.Message }
    }
    $out = $process.StandardOutput.ReadToEndAsync()
    $err = $process.StandardError.ReadToEnd()
    $process.WaitForExit()
    return @{ ExitCode = $process.ExitCode; Out = $out.Result; Err = $err }
}

function Test-VaInstalled {
    param([string]$Name)
    return [bool](Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue)
}

function Test-VaEngineReady {
    param([string]$Engine)
    return ((Invoke-VaEngine -Engine $Engine -Arguments @('info')).ExitCode -eq 0)
}

function Get-VaEngineInfoError {
    param([string]$Engine)
    return (Invoke-VaEngine -Engine $Engine -Arguments @('info')).Err
}

# Podman Desktop runs containers in a virtual machine: start it (make it first).
function Invoke-VaPodmanMachine {
    $machines = Invoke-VaEngine -Engine 'podman' -Arguments @('machine', 'list', '--format', '{{.Name}}')
    Write-VaSay "Starting Podman's virtual machine (first time takes a minute)..."
    if (-not $machines.Out.Trim()) {
        if ((Invoke-VaEngine -Engine 'podman' -Arguments @('machine', 'init')).ExitCode -ne 0) { return $false }
    }
    if ((Invoke-VaEngine -Engine 'podman' -Arguments @('machine', 'start')).ExitCode -ne 0) { return $false }
    return (Test-VaEngineReady 'podman')
}

$script:NoEngineMessage = "VideoAnnotator needs Docker Desktop or Podman Desktop. Install one (see $($script:VaGuide)#install-docker-or-podman), then run this again."

# @{ Engine; BothRunning; Message }: the one asked for; else whichever is
# running; both: the one used last, else Docker; neither: Podman's machine.
function Get-VaEngine {
    param([string]$Saved = '', [string]$Option = '')
    if ($Option) {
        if (Test-VaEngineReady $Option) { return @{ Engine = $Option; BothRunning = $false; Message = '' } }
        if ($Option -eq 'podman' -and (Test-VaInstalled 'podman') -and (Invoke-VaPodmanMachine)) {
            return @{ Engine = 'podman'; BothRunning = $false; Message = '' }
        }
        if (Test-VaInstalled $Option) {
            $why = Get-VaErrorMessage -Engine $Option -Stderr (Get-VaEngineInfoError $Option) -Code 1
            return @{ Engine = ''; BothRunning = $false; Message = $why.Message }
        }
        return @{ Engine = ''; BothRunning = $false; Message = $script:NoEngineMessage }
    }
    $docker = (Test-VaInstalled 'docker') -and (Test-VaEngineReady 'docker')
    $podman = (Test-VaInstalled 'podman') -and (Test-VaEngineReady 'podman')
    if ($docker -and $podman) {
        $engine = 'docker'
        if ($Saved -eq 'podman') { $engine = 'podman' }
        return @{ Engine = $engine; BothRunning = $true; Message = '' }
    }
    if ($docker) { return @{ Engine = 'docker'; BothRunning = $false; Message = '' } }
    if ($podman) { return @{ Engine = 'podman'; BothRunning = $false; Message = '' } }
    if ((Test-VaInstalled 'podman') -and (Invoke-VaPodmanMachine)) {
        return @{ Engine = 'podman'; BothRunning = $false; Message = '' }
    }
    foreach ($engine in @('docker', 'podman')) {
        if (Test-VaInstalled $engine) {
            $why = Get-VaErrorMessage -Engine $engine -Stderr (Get-VaEngineInfoError $engine) -Code 1
            return @{ Engine = ''; BothRunning = $false; Message = $why.Message }
        }
    }
    return @{ Engine = ''; BothRunning = $false; Message = $script:NoEngineMessage }
}

# @{ Message; Detail; Status }: the plain words for an engine failure. Status 0
# when nothing is wrong (already running).
function Get-VaErrorMessage {
    param([string]$Engine, [string]$Stderr = '', [int]$Code = 1, [int]$Port = $script:VaDefaultPort)
    $label = Get-VaEngineLabel $Engine
    $guide = $script:VaGuide
    $text = "$Stderr"
    $message = ''
    $status = 1
    if ($text -match 'already in use by|is already in use') {
        $message = 'VideoAnnotator is already running.'; $status = 0
    } elseif ($text -match 'permission denied while trying to connect') {
        $message = "You don't have permission to use Docker yet. Run: sudo usermod -aG docker `$USER, log out and back in, then run this again. (Or use Podman, which needs no permission.)"
    } elseif ($text -match 'Is the docker daemon running\?|Cannot connect to the Docker daemon|error during connect') {
        $message = "Docker Desktop isn't running. Start it, wait until it says it's running, then run this again."
    } elseif ($text -match 'address already in use|port is already allocated|bind: .*in use') {
        $message = "Something else is using port $Port. Close it, or run: videoannotator-start --port $($Port + 1)"
    } elseif ($text -match 'is not shared from the host|Mounts denied') {
        $path = 'that folder'
        if ($text -match 'The path (\S+) is not shared') { $path = $Matches[1] }
        $message = "Docker Desktop can't see $path yet. Add it in Docker Desktop's Settings, Resources, File sharing (see $guide#docker-desktop-file-sharing), then run this again."
    } elseif ($text -match 'manifest unknown|dial tcp|TLS handshake|no such host|pull access denied|connection refused|i/o timeout') {
        $message = "Couldn't download VideoAnnotator. Check your internet connection and run this again."
    } elseif ($text -match 'OOMKilled|out of memory' -or $Code -eq 137) {
        $message = "VideoAnnotator ran out of memory. Give $label more (see $guide#memory), then run this again."
    }
    $detail = ''
    if (-not $message) {
        $message = "VideoAnnotator couldn't start. Run: videoannotator-start logs"
        $lines = @($text -split "`r?`n" | Where-Object { $_ } | Select-Object -First 3)
        $detail = ($lines | ForEach-Object { "    $_" }) -join "`n"
    }
    return @{ Message = $message; Detail = $detail; Status = $status }
}

function Write-VaError {
    param([hashtable]$Explained)
    Write-VaSay $Explained.Message
    if ($Explained.Detail) { Write-VaSay $Explained.Detail }
}

# ---------------------------------------------------------------------------
# GPU (research R15)
# ---------------------------------------------------------------------------

function Test-VaNvidia {
    if (-not (Test-VaInstalled 'nvidia-smi')) { return $false }
    & nvidia-smi -L *> $null
    return ($LASTEXITCODE -eq 0)
}

function Get-VaCdiDevice {
    return (Invoke-VaEngine -Engine 'podman' -Arguments @('machine', 'ssh', 'nvidia-ctk', 'cdi', 'list')).Out
}

# @{ Flags; Note }. Docker Desktop passes the GPU through WSL whenever the
# NVIDIA driver works; Podman needs the NVIDIA CDI spec in its machine.
function Get-VaGpuFlag {
    param([string]$Engine)
    if (-not (Test-VaNvidia)) { return @{ Flags = @(); Note = '' } }
    if ($Engine -eq 'docker') { return @{ Flags = @('--gpus', 'all'); Note = '' } }
    if ((Get-VaCdiDevice) -match 'nvidia\.com/gpu=') {
        return @{ Flags = @('--device', 'nvidia.com/gpu=all'); Note = '' }
    }
    $label = Get-VaEngineLabel $Engine
    return @{ Flags = @(); Note = "Running without the GPU: $label can't use it yet. To enable it, see $($script:VaGuide)#gpu." }
}

# ---------------------------------------------------------------------------
# The run (contracts/launcher.md, "The run it builds")
# ---------------------------------------------------------------------------

function Get-VaMountArg {
    param([string]$Source, [string]$Target, [string]$Extra = '')
    if ($Source -match '[,"]') {
        # Docker and Podman read --mount as CSV: quote a field holding a comma.
        return ('type=bind,"source=' + $Source.Replace('"', '""') + '",target=' + $Target + $Extra)
    }
    return "type=bind,source=$Source,target=$Target$Extra"
}

function Get-VaRunCommand {
    param(
        [string[]]$Present = @(), [string[]]$Missing = @(),
        [string]$Results, [string]$Port, [string]$Image, [string[]]$GpuFlags = @(),
        [string]$RequestsDir = (Get-VaRequestsDir)
    )
    $run = @('run', '-d', '--name', $script:VaName, '-p', "127.0.0.1:$($Port):$($script:VaInternalPort)")
    $run += @($GpuFlags | Where-Object { $_ })
    foreach ($volume in $script:VaVolumes) { $run += @('-v', "videoannotator-$($volume):/app/$volume") }
    $roots = @()
    $pairs = @()
    $n = 0
    foreach ($share in $Present) {
        if (-not $share) { continue }
        $n++
        $target = ConvertTo-VaContainerPath $share "$n"
        $run += @('--mount', (Get-VaMountArg $share $target ',readonly'))
        $roots += $target
        $pairs += "$target=$share"
    }
    # After the read-only shares, so a results folder inside one stays writable.
    $resultsTarget = ConvertTo-VaContainerPath $Results 'results'
    $run += @('--mount', (Get-VaMountArg $Results $resultsTarget))
    $run += @('--mount', (Get-VaMountArg $RequestsDir '/app/launcher/requests'))
    $pairs += "$resultsTarget=$Results"
    $run += @(
        '-e', ('VIDEOANNOTATOR_INGEST_ROOTS=' + ($roots -join ':')),
        '-e', ('VIDEOANNOTATOR_MISSING_SHARES=' + (@($Missing | Where-Object { $_ }) -join ';')),
        '-e', "VIDEOANNOTATOR_RESULTS_DIR=$resultsTarget",
        '-e', ('VIDEOANNOTATOR_HOST_PATHS=' + ($pairs -join ';')),
        '-e', 'VIDEOANNOTATOR_PUBLISHED_LOCALLY=1',
        '-e', 'VIDEOANNOTATOR_LAUNCHER=1',
        $Image
    )
    # The same for either engine on Windows, and no VIDEOANNOTATOR_RESULTS_OWNER:
    # Docker Desktop's file sharing and Podman's machine give files to the
    # researcher already.
    return , $run
}

# Splits the shares into present and missing, saying which are missing.
function Split-VaShare {
    $script:Present = @()
    $script:Missing = @()
    foreach ($share in $script:Config.shares) {
        if (Test-Path -LiteralPath $share -PathType Container) {
            $script:Present += $share
        } else {
            $script:Missing += $share
            Write-VaSay "Couldn't find $share (an unplugged drive?), so it isn't shared this time."
        }
    }
}

# ---------------------------------------------------------------------------
# Starting, stopping, readiness
# ---------------------------------------------------------------------------

function Test-VaRunning {
    $state = Invoke-VaEngine @('container', 'inspect', '-f', '{{.State.Running}}', $script:VaName)
    return ($state.ExitCode -eq 0 -and $state.Out.Trim() -eq 'true')
}

function Test-VaExist {
    return ((Invoke-VaEngine @('container', 'inspect', $script:VaName)).ExitCode -eq 0)
}

function Invoke-VaPull {
    if ((Invoke-VaEngine @('image', 'inspect', $script:Image)).ExitCode -eq 0) { return $true }
    Write-VaSay 'Downloading VideoAnnotator (first time only, about 1 GB)...'
    $pull = Invoke-VaEngine @('pull', $script:Image)
    if ($pull.ExitCode -eq 0) { return $true }
    Write-VaError (Get-VaErrorMessage -Engine $script:Engine -Stderr $pull.Err -Code 1 -Port $script:Port)
    return $false
}

function Get-VaThisRun {
    return (Get-VaRunCommand -Present $script:Present -Missing $script:Missing `
            -Results $script:Config.results -Port $script:Port -Image $script:Image -GpuFlags $script:GpuFlags)
}

# Starts the container; retries once without the GPU if that is what failed.
function Invoke-VaStart {
    if (Test-VaExist) { [void](Invoke-VaEngine @('rm', '-f', $script:VaName)) }
    $run = Invoke-VaEngine (Get-VaThisRun)
    if ($run.ExitCode -eq 0) { return $true }
    if (@($script:GpuFlags).Count -gt 0) {
        [void](Invoke-VaEngine @('rm', '-f', $script:VaName))
        $script:GpuFlags = @()
        $run = Invoke-VaEngine (Get-VaThisRun)
        if ($run.ExitCode -eq 0) {
            Write-VaSay "Couldn't use the GPU, so VideoAnnotator is running without it. To enable it, see $($script:VaGuide)#gpu."
            return $true
        }
    }
    $explained = Get-VaErrorMessage -Engine $script:Engine -Stderr $run.Err -Code $run.ExitCode -Port $script:Port
    Write-VaError $explained
    if ($explained.Status -eq 0) {
        # Started twice at once: the other one won. Connect to it.
        Open-VaRunning
        exit 0
    }
    return $false
}

function Invoke-VaStop {
    [void](Invoke-VaEngine @('stop', '-t', '30', $script:VaName))
    [void](Invoke-VaEngine @('rm', '-f', $script:VaName))
}

function Get-VaUrl {
    param([string]$Path)
    return "http://127.0.0.1:$($script:Port)$Path"
}

# The body, or $null unless the status is 2xx.
function Invoke-VaHttp {
    param([string]$Url, [string]$Key = '')
    $headers = @{}
    if ($Key) { $headers['Authorization'] = "Bearer $Key" }
    try {
        return (Invoke-WebRequest -Uri $Url -Headers $headers -UseBasicParsing -TimeoutSec 10).Content
    } catch {
        return $null
    }
}

function Test-VaPortAnswer {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $attempt = $client.BeginConnect('127.0.0.1', [int]$script:Port, $null, $null)
        if (-not $attempt.AsyncWaitHandle.WaitOne(1000)) { return $false }
        $client.EndConnect($attempt)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-VaReady {
    Write-VaSay 'Starting...' -NoNewline
    $waited = 0
    while ($waited -lt 300) {
        if ($null -ne (Invoke-VaHttp (Get-VaUrl '/api/v1/system/health'))) {
            Write-VaSay ' ready.'
            return $true
        }
        if (-not (Test-VaRunning)) {
            Write-VaSay ''
            $state = (Invoke-VaEngine @('container', 'inspect', '-f', '{{.State.OOMKilled}} {{.State.ExitCode}}', $script:VaName)).Out.Trim()
            if ($state -like 'true*') {
                Write-VaError (Get-VaErrorMessage -Engine $script:Engine -Stderr 'OOMKilled' -Code 137)
            } else {
                $code = 1
                if ($state -match ' (\d+)$') { $code = [int]$Matches[1] }
                $logs = (Invoke-VaEngine @('logs', '--tail', '20', $script:VaName))
                $tail = @(($logs.Out + "`n" + $logs.Err) -split "`r?`n" | Where-Object { $_ } | Select-Object -Last 3) -join "`n"
                Write-VaError (Get-VaErrorMessage -Engine $script:Engine -Stderr $tail -Code $code)
            }
            return $false
        }
        Start-Sleep -Seconds 2
        $waited += 2
    }
    Write-VaSay ''
    Write-VaSay 'VideoAnnotator is taking a long time to start. Run: videoannotator-start logs'
    return $false
}

# The launcher's administrator key (research R14).
function Get-VaKey {
    if ($script:Config.key -and $null -ne (Invoke-VaHttp (Get-VaUrl '/api/v1/auth/me') $script:Config.key)) {
        return $true
    }
    $made = Invoke-VaEngine @('exec', $script:VaName, 'videoannotator', 'generate-token', '--user', 'researcher@localhost',
        '--username', 'researcher', '--key-name', 'start-up program', '--admin', '--expires-days', '0', '--output', '/tmp/key.json')
    if ($made.ExitCode -ne 0) { return $false }
    $json = (Invoke-VaEngine @('exec', $script:VaName, 'cat', '/tmp/key.json')).Out
    [void](Invoke-VaEngine @('exec', $script:VaName, 'rm', '-f', '/tmp/key.json'))
    if ($json -match '"token":\s*"([^"]+)"') {
        $script:Config.key = $Matches[1]
        return $true
    }
    return $false
}

function Open-VaBrowser {
    Write-VaSay "Opening $(Get-VaUrl '/viewer') in your browser."
    if ($script:NoBrowser) { return }
    Start-Process (Get-VaUrl "/viewer-connect?token=$($script:Config.key)")
}

function Open-VaRunning {
    [void](Get-VaKey)
    Write-VaConfig $script:Config
    Open-VaBrowser
}

# FR-014: what VideoAnnotator can read, and where results go.
function Write-VaSummary {
    $read = @($script:Present) -join ', '
    if (-not $read) { $read = 'no folders yet (run: videoannotator-start share)' }
    Write-VaSay "VideoAnnotator can read: $read. Results: $($script:Config.results)."
}

# ---------------------------------------------------------------------------
# Settings' Stop sharing (research R10)
# ---------------------------------------------------------------------------

function Invoke-VaStopRequest {
    $file = Join-Path (Get-VaRequestsDir) 'stop-sharing.txt'
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { return }
    foreach ($line in [IO.File]::ReadAllLines($file)) {
        $path = $line.Trim()
        if (-not $path) { continue }
        $match = @($script:Config.shares | Where-Object { $_ -ieq $path })
        if ($match.Count -gt 0) {
            $script:Config.shares = @($script:Config.shares | Where-Object { $_ -ine $path })
            Write-VaSay "Stopped sharing $path, as asked in Settings."
        }
    }
    Remove-Item -LiteralPath $file -Force
}

# ---------------------------------------------------------------------------
# Restarting for a change (research R11)
# ---------------------------------------------------------------------------

function Get-VaRunningJobCount {
    $body = Invoke-VaHttp (Get-VaUrl '/api/v1/jobs/?status_filter=running&per_page=1') $script:Config.key
    if ($body -and $body -match '"total":\s*(\d+)') { return [int]$Matches[1] }
    return 0
}

function Invoke-VaRestart {
    param([string]$Announcement)
    if (Test-VaRunning) {
        $jobs = Get-VaRunningJobCount
        if ($jobs -gt 0 -and -not $script:Yes) {
            $what = "$jobs videos are"
            if ($jobs -eq 1) { $what = '1 video is' }
            $answer = Read-VaAnswer "$what being processed. [W]ait for them, or [r]estart now (they'll be marked failed and can be retried)?"
            if ($answer -notmatch '^[Rr]') {
                Write-VaSay 'Waiting for them to finish...'
                while ((Get-VaRunningJobCount) -gt 0) { Start-Sleep -Seconds 10 }
            }
        }
    }
    Write-VaSay $Announcement
    Invoke-VaStop
    return (Invoke-VaLaunch)
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

function Get-VaImage {
    param([string]$Option, [string]$Saved)
    if ($Option) { return $Option }
    if ($script:VaVersion -eq '@VERSION@') {
        if ($Saved) { return $Saved }
        return "$($script:VaImageRepo):latest"
    }
    # Pinned to the launcher's own version: the two always move together.
    return "$($script:VaImageRepo):$($script:VaVersion)"
}

# Pull, start, wait, connect, save.
function Invoke-VaLaunch {
    if ($script:VaVersion -eq '@VERSION@' -and -not $script:ImageOption) {
        Write-VaSay "(development launcher: running $($script:Image))"
    }
    Split-VaShare
    if (Test-VaPortAnswer) {
        Write-VaSay "Something else is using port $($script:Port). Close it, or run: videoannotator-start --port $([int]$script:Port + 1)"
        return $false
    }
    if (-not (Invoke-VaPull)) { return $false }
    $gpu = Get-VaGpuFlag $script:Engine
    $script:GpuFlags = $gpu.Flags
    if ($gpu.Note) { Write-VaSay $gpu.Note }
    [void][IO.Directory]::CreateDirectory($script:Config.results)
    [void][IO.Directory]::CreateDirectory((Get-VaRequestsDir))
    if (-not (Invoke-VaStart)) { return $false }
    if (-not (Wait-VaReady)) { return $false }
    if (-not (Get-VaKey)) {
        Write-VaSay "VideoAnnotator started, but couldn't make its key. Run: videoannotator-start logs"
        return $false
    }
    $script:Config.engine = $script:Engine
    $script:Config.image = $script:Image
    $script:Config.port = "$($script:Port)"
    Write-VaConfig $script:Config
    Write-VaSummary
    Open-VaBrowser
    return $true
}

# Offers an existing compose setup's folders (FR-024).
function Invoke-VaMigrate {
    if ($env:VIDEOS_DIR -or $env:RESULTS_DIR) {
        Write-VaSay 'VideoAnnotator was set up before with:'
        if ($env:VIDEOS_DIR) { Write-VaSay "  videos:  $($env:VIDEOS_DIR)" }
        if ($env:RESULTS_DIR) { Write-VaSay "  results: $($env:RESULTS_DIR)" }
        if (Test-VaYes 'Use your existing VideoAnnotator folders? [Y/n]' 'y') {
            if ($env:RESULTS_DIR) { $script:Config.results = ConvertTo-VaNormalPath $env:RESULTS_DIR }
            if ($env:VIDEOS_DIR) { $script:Config.shares = @(ConvertTo-VaNormalPath $env:VIDEOS_DIR) }
            return $true
        }
    } elseif ((Invoke-VaEngine @('volume', 'inspect', 'videoannotator-database')).ExitCode -eq 0) {
        Write-VaSay 'Your jobs and models will be kept.'
    }
    return $false
}

function Get-VaPickerStart {
    foreach ($name in @('Videos', 'Documents')) {
        $folder = Join-Path $env:USERPROFILE $name
        if (Test-Path -LiteralPath $folder -PathType Container) { return $folder }
    }
    return $env:USERPROFILE
}

# A folder from Windows' folder picker, else typed. $null when cancelled.
function Select-VaFolder {
    param([string]$Question)
    try {
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
        $dialog.Description = $Question
        $dialog.SelectedPath = Get-VaPickerStart
        $dialog.ShowNewFolderButton = $false
        if ($dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { return $null }
        return (ConvertTo-VaNormalPath $dialog.SelectedPath)
    } catch {
        $typed = Read-VaAnswer 'Type the folder path (or press Enter to cancel):'
        if (-not $typed) { return $null }
        return (ConvertTo-VaNormalPath $typed)
    }
}

function Confirm-VaShare {
    param([string]$Folder)
    Write-VaSay ''
    Write-VaSay 'VideoAnnotator will be able to read, but never change:'
    Write-VaSay "  $Folder   (and everything inside it)"
    Write-VaSay 'Results go to:'
    Write-VaSay "  $($script:Config.results)"
    return (Test-VaYes 'Share this folder? [Y/n]' 'y')
}

# First start: which folder are the videos in. $false when cancelled.
function Invoke-VaFirstShare {
    while ($true) {
        Write-VaSay ''
        Write-VaSay 'Which folder are your videos in?'
        $folder = Select-VaFolder 'Which folder are your videos in?'
        if (-not $folder) { return $false }
        $before = @($script:Config.shares)
        if ((Add-VaShare $folder) -eq 0) {
            if (Confirm-VaShare $folder) { return $true }
            $script:Config.shares = $before
        }
    }
}

function Use-VaEngine {
    $found = Get-VaEngine -Saved $script:Config.engine -Option $script:EngineOption
    if (-not $found.Engine) {
        Write-VaSay $found.Message
        exit 1
    }
    $script:Engine = $found.Engine
    $script:BothRunning = $found.BothRunning
}

function Invoke-VaCommandStart {
    $first = -not (Test-Path -LiteralPath (Get-VaSettingsPath) -PathType Leaf)
    Use-VaEngine
    if ($first -or $script:BothRunning -or ($script:Config.engine -and $script:Config.engine -ne $script:Engine)) {
        Write-VaSay "Starting VideoAnnotator with $(Get-VaEngineLabel $script:Engine)."
    }
    if (Test-VaRunning) {
        Write-VaSay 'VideoAnnotator is already running.'
        Open-VaRunning
        exit 0
    }
    Invoke-VaStopRequest
    if ($first) { [void](Invoke-VaMigrate) }
    if ($script:ResultsOption) { $script:Config.results = ConvertTo-VaNormalPath $script:ResultsOption }
    if (-not $script:Config.results) { $script:Config.results = Join-Path $env:USERPROFILE 'VideoAnnotator' }
    if ($script:ShareOptions.Count -gt 0) {
        foreach ($share in $script:ShareOptions) { [void](Add-VaShare $share) }
    } elseif ($first -and @($script:Config.shares).Count -eq 0) {
        if (-not (Invoke-VaFirstShare)) {
            Write-VaSay 'Nothing was shared.'
            exit 2
        }
    }
    [void][IO.Directory]::CreateDirectory($script:Config.results)
    Write-VaConfig $script:Config
    if (-not (Invoke-VaLaunch)) { exit 1 }
}

function Invoke-VaCommandShare {
    param([string]$Path)
    Use-VaEngine
    if (-not $script:Config.results) { $script:Config.results = Join-Path $env:USERPROFILE 'VideoAnnotator' }
    $folder = $Path
    if (-not $folder) {
        $folder = Select-VaFolder 'Which folder do you want to share?'
        if (-not $folder) { exit 2 }
    }
    $added = Add-VaShare $folder
    if ($added -ne 0) { exit $added }
    $folder = ConvertTo-VaNormalPath $folder
    if (-not $Path -and -not (Confirm-VaShare $folder)) { exit 2 }
    Write-VaConfig $script:Config
    if (-not (Invoke-VaRestart "Restarting VideoAnnotator to share $folder...")) { exit 1 }
}

function Invoke-VaCommandUnshare {
    param([string]$Path)
    Use-VaEngine
    $shares = @($script:Config.shares)
    if ($shares.Count -eq 0) {
        Write-VaSay 'No folders are shared.'
        exit 0
    }
    if ($Path) {
        $folder = ConvertTo-VaNormalPath $Path
    } else {
        Write-VaSay 'Shared folders:'
        for ($i = 0; $i -lt $shares.Count; $i++) { Write-VaSay "  $($i + 1). $($shares[$i])" }
        $number = Read-VaAnswer 'Which one should VideoAnnotator stop reading? (number, or Enter to cancel)'
        if (-not $number) { exit 2 }
        $folder = ''
        if ($number -match '^\d+$' -and [int]$number -ge 1 -and [int]$number -le $shares.Count) {
            $folder = $shares[[int]$number - 1]
        }
    }
    $match = @($shares | Where-Object { $_ -ieq $folder })
    if (-not $folder -or $match.Count -eq 0) {
        Write-VaSay "$folder isn't shared."
        exit 1
    }
    $script:Config.shares = @($shares | Where-Object { $_ -ine $folder })
    Write-VaConfig $script:Config
    if (-not (Invoke-VaRestart "Restarting VideoAnnotator to stop sharing $folder...")) { exit 1 }
}

function Invoke-VaCommandList {
    Write-VaSay 'Shared folders (read-only):'
    if (@($script:Config.shares).Count -eq 0) { Write-VaSay '  (none)' }
    foreach ($share in $script:Config.shares) { Write-VaSay "  $share" }
    $results = $script:Config.results
    if (-not $results) { $results = Join-Path $env:USERPROFILE 'VideoAnnotator' }
    Write-VaSay 'Results folder:'
    Write-VaSay "  $results"
    $engine = $script:Config.engine
    if (-not $engine) { $engine = 'docker' }
    Write-VaSay "Engine: $(Get-VaEngineLabel $engine)"
    $image = $script:Config.image
    if (-not $image) { $image = $script:Image }
    Write-VaSay "Image:  $image"
}

# Whether version A is newer than version B (numbers, not text).
function Test-VaNewer {
    param([string]$A, [string]$B)
    $x = @($A.TrimStart('v').Split('.') | ForEach-Object { [int]($_ -replace '\D.*$', '0') })
    $y = @($B.TrimStart('v').Split('.') | ForEach-Object { [int]($_ -replace '\D.*$', '0') })
    for ($i = 0; $i -lt [Math]::Max($x.Count, $y.Count); $i++) {
        $p = 0; $q = 0
        if ($i -lt $x.Count) { $p = $x[$i] }
        if ($i -lt $y.Count) { $q = $y[$i] }
        if ($p -gt $q) { return $true }
        if ($p -lt $q) { return $false }
    }
    return $false
}

function Get-VaLatestRelease {
    $body = Invoke-VaHttp "https://api.github.com/repos/$($script:VaGithub)/releases/latest"
    if ($body -and $body -match '"tag_name":\s*"([^"]+)"') { return $Matches[1] }
    return ''
}

function Invoke-VaCommandUpdate {
    param([switch]$Continue)
    Use-VaEngine
    if (-not $Continue -and $script:VaVersion -ne '@VERSION@') {
        $latest = Get-VaLatestRelease
        if ($latest -and (Test-VaNewer $latest $script:VaVersion)) {
            Write-VaSay "Updating the start-up program to $latest..."
            try {
                $installer = Invoke-VaHttp "https://github.com/$($script:VaGithub)/releases/download/$latest/install.ps1"
                if (-not $installer) { throw 'no installer' }
                $env:VA_RELEASE = $latest
                & ([scriptblock]::Create($installer)) -NoShortcut -Quiet
            } catch {
                Write-VaSay "Couldn't download VideoAnnotator. Check your internet connection and run this again."
                exit 1
            }
            $next = Join-Path $env:LOCALAPPDATA 'VideoAnnotator\videoannotator-start.ps1'
            & powershell -NoProfile -ExecutionPolicy Bypass -File $next update --continue
            exit $LASTEXITCODE
        }
    }
    $before = (Invoke-VaEngine @('image', 'inspect', '-f', '{{.Id}}', $script:Image)).Out.Trim()
    $pull = Invoke-VaEngine @('pull', $script:Image)
    if ($pull.ExitCode -ne 0) {
        Write-VaError (Get-VaErrorMessage -Engine $script:Engine -Stderr $pull.Err -Code 1)
        exit 1
    }
    $after = (Invoke-VaEngine @('image', 'inspect', '-f', '{{.Id}}', $script:Image)).Out.Trim()
    $current = (Invoke-VaEngine @('container', 'inspect', '-f', '{{.Image}}', $script:VaName)).Out.Trim()
    $saved = $script:Config.image
    if (-not $saved) { $saved = $script:Image }
    if ($before -eq $after -and (-not $current -or $current -eq $after) -and $saved -eq $script:Image) {
        Write-VaSay 'VideoAnnotator is up to date.'
        exit 0
    }
    $script:Config.image = $script:Image
    Write-VaConfig $script:Config
    if (-not (Invoke-VaRestart 'Restarting VideoAnnotator with the new version...')) { exit 1 }
    Write-VaSay 'Updated. Your folders, results, models and installed pipelines are kept.'
}

function Write-VaUsage {
    Write-VaSay @'
Usage: videoannotator-start [command] [options]

Commands:
  (none)          Start VideoAnnotator, or open it if it is already running
  share [PATH]    Share another folder (read-only), then restart
  unshare [PATH]  Stop sharing a folder, then restart
  list            Show shared folders, results folder, engine and image
  stop            Stop VideoAnnotator
  update          Update VideoAnnotator (your folders, results and models are kept)
  logs            Show VideoAnnotator's recent log

Options:
  --share PATH    Share this folder (repeatable)
  --results PATH  Write results here (default: your VideoAnnotator folder)
  --engine docker|podman
  --yes           Answer yes to every question
  --no-browser    Don't open the browser
  --port N        Use this port on 127.0.0.1 (default: 18011)
  --image REF     Run this image
'@
}

function Invoke-VaMain {
    param([string[]]$Arguments)
    $command = ''
    $argument = ''
    $continue = $false
    $script:ShareOptions = @()
    $script:ResultsOption = ''
    $script:EngineOption = ''
    $script:ImageOption = ''
    $portOption = ''
    $script:Yes = $false
    $script:NoBrowser = $false
    for ($i = 0; $i -lt $Arguments.Count; $i++) {
        $a = $Arguments[$i]
        $value = $null
        if ($i + 1 -lt $Arguments.Count) { $value = $Arguments[$i + 1] }
        switch -Regex ($a) {
            '^--share$' { $script:ShareOptions += $value; $i++; continue }
            '^--results$' { $script:ResultsOption = $value; $i++; continue }
            '^--engine$' { $script:EngineOption = $value; $i++; continue }
            '^--image$' { $script:ImageOption = $value; $i++; continue }
            '^--port$' { $portOption = $value; $i++; continue }
            '^(--yes|-y)$' { $script:Yes = $true; continue }
            '^--no-browser$' { $script:NoBrowser = $true; continue }
            '^--continue$' { $continue = $true; continue }
            '^(-h|--help|/\?)$' { Write-VaUsage; exit 0 }
            '^-' { Write-VaSay "Unknown option: $a"; Write-VaUsage; exit 1 }
            default { if (-not $command) { $command = $a } else { $argument = $a } }
        }
    }
    if ($script:EngineOption -and $script:EngineOption -notin @('docker', 'podman')) { Write-VaUsage; exit 1 }

    $script:Config = Read-VaConfig
    $script:Port = $script:VaDefaultPort
    if ($script:Config.port) { $script:Port = [int]$script:Config.port }
    if ($portOption) { $script:Port = [int]$portOption }
    $script:Image = Get-VaImage -Option $script:ImageOption -Saved $script:Config.image
    $script:GpuFlags = @()

    if (-not $command) { $command = 'start' }
    switch ($command) {
        'start' { Invoke-VaCommandStart }
        'share' { Invoke-VaCommandShare $argument }
        'unshare' { Invoke-VaCommandUnshare $argument }
        'list' { Invoke-VaCommandList }
        'stop' {
            Use-VaEngine
            Invoke-VaStop
            Write-VaSay 'VideoAnnotator has stopped.'
        }
        'logs' {
            Use-VaEngine
            & $script:Engine logs --tail 200 $script:VaName
        }
        'update' { Invoke-VaCommandUpdate -Continue:$continue }
        'help' { Write-VaUsage }
        default { Write-VaSay "Unknown command: $command"; Write-VaUsage; exit 1 }
    }
    exit 0
}

if ($env:VA_SOURCE_ONLY -ne '1') { Invoke-VaMain $args }
