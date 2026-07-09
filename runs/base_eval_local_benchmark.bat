@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Capture the script/repo location up front: SHIFT during arg parsing
rem mutates %0, which would corrupt %~dp0 if read later.
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

set "PYTHON="
set "EVAL=core"
set "MAX_PER_TASK=-1"
set "DEVICE_TYPE="
set "NANOCHAT_BASE_DIR_LOCAL="
set "CHECKPOINTS_DIR="
set "STOP_ON_ERROR=0"
set "MODELS="
set "STEP="
set "DRY_RUN=0"
set "OVERALL_RC=0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="-h" goto usage_ok
if /I "%~1"=="--help" goto usage_ok
if /I "%~1"=="--python" (
    if "%~2"=="" goto usage_err
    set "PYTHON=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--eval" (
    if "%~2"=="" goto usage_err
    set "EVAL=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--max-per-task" (
    if "%~2"=="" goto usage_err
    set "MAX_PER_TASK=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--device-type" (
    if "%~2"=="" goto usage_err
    set "DEVICE_TYPE=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--step" (
    if "%~2"=="" goto usage_err
    set "STEP=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--nanochat-base-dir" (
    if "%~2"=="" goto usage_err
    set "NANOCHAT_BASE_DIR_LOCAL=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--checkpoints-dir" (
    if "%~2"=="" goto usage_err
    set "CHECKPOINTS_DIR=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--stop-on-error" (
    set "STOP_ON_ERROR=1"
    shift
    goto parse_args
)
if /I "%~1"=="--models" (
    if "%~2"=="" goto usage_err
    set "MODELS=%~2"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--dry-run" (
    set "DRY_RUN=1"
    shift
    goto parse_args
)
echo Unknown argument: %~1
goto usage_err

:args_done
rem Pick a Python launcher. Prefer `uv run python` (the repo's managed
rem environment) when uv is available, otherwise fall back to `python`.
if not defined PYTHON (
    where uv >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON=uv run python"
    ) else (
        set "PYTHON=python"
    )
)

rem Default the nanochat base dir to the repo root so that both
rem ./base_checkpoints and ./tokenizer resolve locally.
if not defined NANOCHAT_BASE_DIR_LOCAL (
    if defined NANOCHAT_BASE_DIR (
        set "NANOCHAT_BASE_DIR_LOCAL=%NANOCHAT_BASE_DIR%"
    ) else (
        set "NANOCHAT_BASE_DIR_LOCAL=%REPO_ROOT%"
    )
)
set "NANOCHAT_BASE_DIR=%NANOCHAT_BASE_DIR_LOCAL%"

rem Default the checkpoints dir to <base_dir>\base_checkpoints.
if not defined CHECKPOINTS_DIR (
    set "CHECKPOINTS_DIR=%NANOCHAT_BASE_DIR%\base_checkpoints"
)

if not exist "%CHECKPOINTS_DIR%" (
    echo Checkpoints directory not found: "%CHECKPOINTS_DIR%"
    exit /b 1
)

rem If no explicit model tags are given, discover every model subfolder
rem under the checkpoints directory (each subfolder is one model tag).
if not defined MODELS (
    for /d %%D in ("%CHECKPOINTS_DIR%\*") do (
        if defined MODELS (
            set "MODELS=!MODELS! %%~nxD"
        ) else (
            set "MODELS=%%~nxD"
        )
    )
)
set "MODELS=%MODELS:,= %"

if not defined MODELS (
    echo No models found under "%CHECKPOINTS_DIR%".
    exit /b 1
)

echo Base dir       : %NANOCHAT_BASE_DIR%
echo Checkpoints    : %CHECKPOINTS_DIR%
echo Models to eval : %MODELS%

pushd "%REPO_ROOT%" >nul || (
    echo Failed to enter repository root: "%REPO_ROOT%"
    exit /b 1
)

for %%M in (%MODELS%) do (
    echo [%DATE% %TIME%] model running - %%M
    set "CMD=%PYTHON% -m scripts.base_eval --eval %EVAL% --model-tag %%M --max-per-task %MAX_PER_TASK%"
    if defined STEP set "CMD=!CMD! --step %STEP%"
    if defined DEVICE_TYPE set "CMD=!CMD! --device-type %DEVICE_TYPE%"

    if "!DRY_RUN!"=="1" (
        echo [dry-run] !CMD!
        set "RC=0"
    ) else (
        call !CMD!
        set "RC=!ERRORLEVEL!"
    )

    if not "!RC!"=="0" (
        echo [%DATE% %TIME%] model failed - %%M ^(exit !RC!^)
        if "!OVERALL_RC!"=="0" set "OVERALL_RC=!RC!"
        if "!STOP_ON_ERROR!"=="1" (
            popd >nul
            exit /b !RC!
        )
    ) else (
        echo [%DATE% %TIME%] model done - %%M
    )
)

popd >nul
exit /b %OVERALL_RC%

:usage_ok
echo Usage:
echo   runs\base_eval_local_benchmark.bat [options]
echo.
echo Discovers local nanochat model checkpoints under the base_checkpoints
echo folder and runs scripts.base_eval against each one, one at a time.
echo.
echo Options:
echo   --python ^<exe^>              Python launcher. Default: "uv run python" if uv
echo                                 is available, else "python"
echo   --eval ^<modes^>              Eval mode(s) for scripts.base_eval. Default: core
echo   --max-per-task ^<n^>          Max examples per CORE task. Default: -1
echo   --device-type ^<type^>        cuda^|cpu^|mps ^(empty = autodetect^)
echo   --step ^<n^>                  Model step to load ^(default = last^)
echo   --nanochat-base-dir ^<path^>  Base dir for checkpoints/tokenizer. Default: repo root
echo   --checkpoints-dir ^<path^>    Folder holding model tag subfolders.
echo                                 Default: ^<base-dir^>\base_checkpoints
echo   --models "^<t1 t2 ...^>"      Space- or comma-separated model tags.
echo                                 Default: all subfolders under checkpoints dir
echo   --stop-on-error               Stop on first failing model
echo   --dry-run                     Print commands without executing
echo   -h, --help                    Show this help
echo.
echo Note: the tokenizer must exist at ^<base-dir^>\tokenizer\ for nanochat
echo models to run ^(tokenizer.pkl + token_bytes.pt^).
exit /b 0

:usage_err
call :usage_ok
exit /b 1
