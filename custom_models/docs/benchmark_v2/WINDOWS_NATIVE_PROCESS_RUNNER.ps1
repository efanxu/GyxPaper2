Set-StrictMode -Version 2.0

function ConvertTo-GyxWindowsCommandLineArgument {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Argument
    )

    if ($Argument.Length -gt 0 -and $Argument -notmatch '[\s"]') {
        return $Argument
    }

    # ProcessStartInfo.Arguments is one Windows command-line string under
    # Windows PowerShell 5.1. Quote exactly as CommandLineToArgvW expects.
    $builder = New-Object System.Text.StringBuilder
    [void]$builder.Append('"')
    $backslashes = 0
    foreach ($character in $Argument.ToCharArray()) {
        if ($character -eq '\') {
            $backslashes++
            continue
        }
        if ($character -eq '"') {
            [void]$builder.Append(('\' * (($backslashes * 2) + 1)))
            [void]$builder.Append('"')
            $backslashes = 0
            continue
        }
        if ($backslashes -gt 0) {
            [void]$builder.Append(('\' * $backslashes))
            $backslashes = 0
        }
        [void]$builder.Append($character)
    }
    if ($backslashes -gt 0) {
        [void]$builder.Append(('\' * ($backslashes * 2)))
    }
    [void]$builder.Append('"')
    return $builder.ToString()
}

function Join-GyxWindowsCommandLine {
    param([string[]]$Arguments = @())

    return (($Arguments | ForEach-Object {
        ConvertTo-GyxWindowsCommandLineArgument -Argument ([string]$_)
    }) -join ' ')
}

function Invoke-GyxNativeProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory
    )

    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $FilePath
    $startInfo.Arguments = Join-GyxWindowsCommandLine -Arguments $Arguments
    $startInfo.WorkingDirectory = $WorkingDirectory
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    $startInfo.StandardOutputEncoding = $utf8WithoutBom
    $startInfo.StandardErrorEncoding = $utf8WithoutBom
    $startInfo.EnvironmentVariables['PYTHONUTF8'] = '1'
    $startInfo.EnvironmentVariables['PYTHONIOENCODING'] = 'utf-8'

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            throw "Failed to start native process: $FilePath"
        }
        # Drain both redirected pipes asynchronously before waiting so a large
        # stderr stream cannot deadlock the child process.
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $stdout = [string]$stdoutTask.Result
        $stderr = [string]$stderrTask.Result
        $exitCode = [int]$process.ExitCode
    } finally {
        $process.Dispose()
    }

    return [pscustomobject]@{
        ExitCode = $exitCode
        Stdout = $stdout
        Stderr = $stderr
    }
}

function Invoke-GyxPythonGate {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Python,
        [Parameter(Mandatory = $true)]
        [string]$Gate,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$LogRoot,
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [int[]]$AllowedExitCodes = @(0),
        [string]$ReportPath
    )

    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    $attemptToken = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '-' + [Guid]::NewGuid().ToString('N')
    $logPath = Join-Path $LogRoot ($Label + '.' + $attemptToken + '.stdout.log')
    $stderrPath = Join-Path $LogRoot ($Label + '.' + $attemptToken + '.stderr.log')
    $childArguments = @($Gate) + @($Arguments)
    $native = Invoke-GyxNativeProcess `
        -FilePath $Python `
        -Arguments $childArguments `
        -WorkingDirectory $WorkingDirectory

    $utf8WithoutBom = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($logPath, [string]$native.Stdout, $utf8WithoutBom)
    [IO.File]::WriteAllText($stderrPath, [string]$native.Stderr, $utf8WithoutBom)

    if (-not [string]::IsNullOrEmpty([string]$native.Stdout)) {
        Write-Host -NoNewline ([string]$native.Stdout)
        if (-not ([string]$native.Stdout).EndsWith("`n")) {
            Write-Host ''
        }
    }
    if (-not [string]::IsNullOrEmpty([string]$native.Stderr)) {
        Write-Host -ForegroundColor DarkYellow -NoNewline ([string]$native.Stderr)
        if (-not ([string]$native.Stderr).EndsWith("`n")) {
            Write-Host ''
        }
    }

    # Parse evidence before applying the exit-code policy. A suite exit 1 or 4
    # is protocol state, not a transport failure, and its report must survive.
    $json = $null
    if (-not [string]::IsNullOrWhiteSpace($ReportPath) -and (Test-Path -LiteralPath $ReportPath -PathType Leaf)) {
        try {
            $json = Get-Content -LiteralPath $ReportPath -Raw -Encoding UTF8 | ConvertFrom-Json
        } catch {
            $json = $null
        }
    }
    if ($null -eq $json -and -not [string]::IsNullOrWhiteSpace([string]$native.Stdout)) {
        try {
            $json = ([string]$native.Stdout) | ConvertFrom-Json
        } catch {
            $json = $null
        }
    }

    $result = [pscustomobject]@{
        Label = $Label
        ExitCode = [int]$native.ExitCode
        Stdout = [string]$native.Stdout
        Stderr = [string]$native.Stderr
        Text = [string]$native.Stdout
        Json = $json
        LogPath = $logPath
        StderrPath = $stderrPath
    }
    if ($AllowedExitCodes -notcontains [int]$native.ExitCode) {
        $parsedStatus = '<unparsed>'
        if ($null -ne $json -and $null -ne $json.status) {
            $parsedStatus = [string]$json.status
        }
        throw "$Label failed with exit code $($native.ExitCode); parsed_status=$parsedStatus; stdout_log=$logPath; stderr_log=$stderrPath"
    }
    return $result
}
