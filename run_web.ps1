$ErrorActionPreference = "Stop"

# Windows setup equivalent of the shell bootstrap block in my_run.sh.

function Import-CmdEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BatchFile,
        [string]$BatchArgs = ""
    )

    $cmdLine = if ([string]::IsNullOrWhiteSpace($BatchArgs)) {
        "`"$BatchFile`" >nul && set"
    }
    else {
        "`"$BatchFile`" $BatchArgs >nul && set"
    }

    $envDump = cmd /c $cmdLine
    foreach ($line in $envDump) {
        if ($line -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

function Ensure-MsvcLinker {
    if (Get-Command link.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $vsInstallPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
        if ($vsInstallPath) {
            $vsDevCmd = Join-Path $vsInstallPath "Common7\Tools\VsDevCmd.bat"
            if (Test-Path $vsDevCmd) {
                # Import VS developer shell env vars into this PowerShell process.
                Import-CmdEnvironment -BatchFile $vsDevCmd -BatchArgs "-no_logo -arch=x64"
            }
        }
    }

    if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
        throw "MSVC linker (link.exe) not found. Install Visual Studio Build Tools with 'Desktop development with C++' and rerun this script."
    }
}

# 1) Ensure uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
}

# 2) Ensure virtual environment exists
if (-not (Test-Path ".venv")) {
    uv venv
}

# 3) Sync dependencies
uv sync

# 4) Activate venv for this PowerShell session
. .\.venv\Scripts\Activate.ps1

# 5) Provide default WANDB_RUN if not set
if ([string]::IsNullOrWhiteSpace($env:WANDB_RUN)) {
    $env:WANDB_RUN = "dummy"
}

# 6) Ensure Rust toolchain exists
if (-not (Get-Command rustup -ErrorAction SilentlyContinue)) {
    $rustupExe = Join-Path $env:TEMP "rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupExe
    & $rustupExe -y
}

# 7) Ensure Cargo bin is available in current session PATH
$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ((Test-Path $cargoBin) -and ($env:PATH -notlike "*$cargoBin*")) {
    $env:PATH = "$cargoBin;$env:PATH"
}

# 8) Build/install Rust Python extension in develop mode
Ensure-MsvcLinker
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# 9) Run web server
uv run python -m scripts.chat_web