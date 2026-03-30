# Builds the Release x64 solution with MSBuild and runs the generated executable from the repo root.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$solutionPath = Join-Path $repoRoot 'nn.sln'
$binaryPath = Join-Path $repoRoot 'bin\nn.exe'

function Get-MSBuildPath {
    $vswherePath = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'

    if (Test-Path $vswherePath) {
        $installationPath = & $vswherePath -latest -products * -requires Microsoft.Component.MSBuild -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installationPath) {
            $candidate = Join-Path $installationPath 'MSBuild\Current\Bin\amd64\MSBuild.exe'
            if (Test-Path $candidate) {
                return $candidate
            }

            $candidate = Join-Path $installationPath 'MSBuild\Current\Bin\MSBuild.exe'
            if (Test-Path $candidate) {
                return $candidate
            }
        }
    }

    $command = Get-Command msbuild.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    throw 'MSBuild.exe was not found. Install Visual Studio Build Tools or add MSBuild to PATH.'
}

$msbuildPath = Get-MSBuildPath

Write-Host "Building Release|x64 with $msbuildPath"
& $msbuildPath $solutionPath '/p:Configuration=Release' '/p:Platform=x64'

if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $binaryPath)) {
    throw "Expected executable was not produced: $binaryPath"
}

Write-Host "Running $binaryPath"
Push-Location $repoRoot
try {
    & $binaryPath
}
finally {
    Pop-Location
}