# The PowerShell launcher against tests/launcher/cases.json: every row whose
# os is "windows" or "any" (launcher/README.md). Pester 5.

BeforeDiscovery {
    $cases = Get-Content -Raw -Path (Join-Path $PSScriptRoot 'cases.json') | ConvertFrom-Json
    function Get-Row([string]$Group) {
        @($cases.$Group | Where-Object { $_.os -in @('windows', 'any') } | ForEach-Object { @{ Row = $_; Name = $_.name } })
    }
    $paths = Get-Row 'paths'
    $normalise = Get-Row 'normalise'
    $classify = Get-Row 'classify'
    $settings = Get-Row 'settings'
    $runs = Get-Row 'run'
    $engines = Get-Row 'engine'
    $gpus = Get-Row 'gpu'
    $messages = Get-Row 'messages'
}

BeforeAll {
    $env:VA_SOURCE_ONLY = '1'
    . (Join-Path (Join-Path (Join-Path $PSScriptRoot '..') '..') (Join-Path 'launcher' 'videoannotator-start.ps1'))
    function Get-Field($Object, [string]$Name, $Default = '') {
        $property = $Object.PSObject.Properties[$Name]
        if ($null -eq $property) { return $Default }
        return $property.Value
    }
}

Describe 'paths' {
    It '<Name>' -ForEach $paths {
        ConvertTo-VaContainerPath $Row.input.path "$($Row.input.n)" | Should -BeExactly $Row.expect.container
    }
}

Describe 'normalise' {
    It '<Name>' -ForEach $normalise {
        ConvertTo-VaNormalPath -Path $Row.input.path -HomeDir $Row.input.home | Should -BeExactly $Row.expect.path
    }
}

Describe 'classify' {
    It '<Name>' -ForEach $classify {
        $result = Get-VaClassification -Path $Row.input.path -Shares @($Row.input.shares) `
            -Results $Row.input.results -HomeDir $Row.input.home
        $result | Should -BeExactly $Row.expect.result
    }
}

Describe 'settings' {
    It '<Name>' -ForEach $settings {
        Get-VaSettingsPath -AppData $Row.input.appdata | Should -BeExactly $Row.expect.path
    }

    It 'round-trips, keeping order and spaces' {
        $file = Join-Path $TestDrive 'start.conf'
        $config = @{
            engine = 'podman'; image = 'ghcr.io/infantlab/videoannotator:1.6.0'
            results = 'C:\Users\ada\My Results'; key = 'va_secret'; port = '18012'
            shares = @('C:\Users\ada\Studies\Day 1', 'E:\Drive 2', 'C:\Users\ada\Etudes')
        }
        Write-VaConfig -Config $config -Path $file
        $read = Read-VaConfig -Path $file
        foreach ($name in @('engine', 'image', 'results', 'key', 'port')) { $read[$name] | Should -BeExactly $config[$name] }
        $read.shares | Should -Be $config.shares
        Test-Path (Join-Path $TestDrive 'requests') | Should -BeTrue
    }

    It 'reads nothing from no file' {
        $read = Read-VaConfig -Path (Join-Path $TestDrive 'none.conf')
        @($read.shares).Count | Should -Be 0
    }
}

Describe 'run' {
    It '<Name>' -ForEach $runs {
        $i = $Row.input
        $requests = Get-VaRequestsDir -AppData $i.appdata
        $gpu = @()
        if ($i.gpu) { $gpu = $i.gpu.Split(' ') }
        $run = Get-VaRunCommand -Present @($i.shares) -Missing @($i.missing) `
            -Results $i.results -Port $i.port -Image $i.image -GpuFlags $gpu -RequestsDir $requests
        ($run -join "`n") | Should -BeExactly (@($Row.expect.args) -join "`n")
    }
}

Describe 'engine' {
    It '<Name>' -ForEach $engines {
        $i = $Row.input
        $script:States = @{ docker = $i.docker; podman = $i.podman }
        $script:InfoError = Get-Field $i 'stderr'
        Mock Test-VaInstalled { param($Name) $script:States[$Name] -ne 'absent' }
        Mock Test-VaEngineReady { param($Engine) $script:States[$Engine] -eq 'running' }
        Mock Get-VaEngineInfoError { $script:InfoError }
        Mock Invoke-VaPodmanMachine { $false }
        $found = Get-VaEngine -Saved $i.saved -Option $i.option
        $expected = Get-Field $Row.expect 'engine'
        if ($expected) {
            $found.Engine | Should -BeExactly $expected
            if (Get-Field $Row.expect 'announced' $false) { $found.BothRunning | Should -BeTrue }
        } else {
            $found.Engine | Should -BeNullOrEmpty
            $found.Message | Should -BeExactly $Row.expect.message
        }
    }

    It "starts Podman's machine when it is stopped" {
        Mock Test-VaInstalled { param($Name) $Name -eq 'podman' }
        Mock Test-VaEngineReady { $false }
        Mock Invoke-VaPodmanMachine { $true }
        (Get-VaEngine).Engine | Should -BeExactly 'podman'
    }
}

Describe 'gpu' {
    It '<Name>' -ForEach $gpus {
        $i = $Row.input
        $script:Nvidia = [bool]$i.nvidia
        $script:Cdi = $i.cdi
        Mock Test-VaNvidia { $script:Nvidia }
        Mock Get-VaCdiDevice { $script:Cdi }
        $gpu = Get-VaGpuFlag -Engine $i.engine
        ($gpu.Flags -join ' ') | Should -BeExactly $Row.expect.flags
        if ($Row.expect.note) { $gpu.Note | Should -Match "^Running without the GPU: .* can't use it yet\." }
        else { $gpu.Note | Should -BeNullOrEmpty }
    }
}

Describe 'messages' {
    It '<Name>' -ForEach $messages {
        $i = $Row.input
        $explained = Get-VaErrorMessage -Engine $i.engine -Stderr $i.stderr -Code ([int]$i.code) -Port 18011
        $explained.Message | Should -BeExactly $Row.expect.message
        $explained.Status | Should -Be $Row.expect.status
        $detail = Get-Field $Row.expect 'detail'
        if ($detail) { $explained.Detail | Should -BeExactly $detail }
    }
}

Describe 'arguments reach the engine as Windows programs read them' {
    It 'quotes spaces, quotes and trailing backslashes' {
        ConvertTo-VaArgString @('run', 'C:\My Videos\', 'type=bind,"source=E:\a,b",target=/host/1', '') |
            Should -BeExactly 'run "C:\My Videos\\" "type=bind,\"source=E:\a,b\",target=/host/1" ""'
    }
}

Describe 'versions' {
    It 'compare by number' {
        Test-VaNewer 'v1.10.0' '1.9.2' | Should -BeTrue
        Test-VaNewer '1.6.0' '1.6.0' | Should -BeFalse
        Test-VaNewer '1.5.9' 'v1.6.0' | Should -BeFalse
    }
}

Describe 'shares and stop requests' {
    BeforeEach {
        $env:APPDATA = Join-Path $TestDrive 'AppData'
        $root = Join-Path $TestDrive 'disk'
        $studies = Join-Path $root 'Studies'
        $second = Join-Path $root 'Second'
        [void](New-Item -ItemType Directory -Force -Path $studies, $second, (Get-VaRequestsDir))
        $script:Config = @{ engine = ''; image = ''; results = (Join-Path $root 'Results'); key = ''; port = ''; shares = @($studies, $second) }
    }

    It 'applies stop requests from Settings, ignores anything else, and deletes the file' {
        $file = Join-Path (Get-VaRequestsDir) 'stop-sharing.txt'
        Set-Content -Path $file -Value @($studies.ToUpperInvariant(), 'C:\Windows')
        Mock Write-VaSay {}
        Invoke-VaStopRequest
        $script:Config.shares | Should -Be @($second)
        Test-Path $file | Should -BeFalse
        Should -Invoke Write-VaSay -ParameterFilter { $Text -like 'Stopped sharing *, as asked in Settings.' } -Times 1
    }

    # Windows paths only: TestDrive is a POSIX path when pwsh runs these on Linux.
    It 'a containing folder replaces the shares inside it' -Skip:([Environment]::OSVersion.Platform -ne 'Win32NT') {
        $script:Yes = $false
        (Add-VaShare $root) | Should -Be 0
        $script:Config.shares | Should -Be @($root)
    }

    It 'a missing share is named and left out of the run' {
        $script:Config.shares = @($studies, 'Q:\Unplugged')
        Mock Write-VaSay {}
        Split-VaShare
        $script:Present | Should -Be @($studies)
        $script:Missing | Should -Be @('Q:\Unplugged')
        Should -Invoke Write-VaSay -ParameterFilter { $Text -eq "Couldn't find Q:\Unplugged (an unplugged drive?), so it isn't shared this time." }
    }
}
