@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "PYTHON=python"
set "EVAL=core"
set "MAX_PER_TASK=-1"
set "DEVICE_TYPE="
set "NANOCHAT_BASE_DIR_LOCAL="
set "STOP_ON_ERROR=0"
set "MODELS="
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
if /I "%~1"=="--nanochat-base-dir" (
    if "%~2"=="" goto usage_err
    set "NANOCHAT_BASE_DIR_LOCAL=%~2"
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
if not defined MODELS (
    set "MODELS=openai-community/gpt2 Polygl0t/Tucano2-0.6B-Base Polygl0t/Tucano2-qwen-1.5B-Base TucanoBR/Tucano-630m TucanoBR/Tucano-1b1"
)
set "MODELS=%MODELS:,= %"

if defined NANOCHAT_BASE_DIR_LOCAL (
    set "NANOCHAT_BASE_DIR=%NANOCHAT_BASE_DIR_LOCAL%"
)

for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"
pushd "%REPO_ROOT%" >nul || (
    echo Failed to enter repository root: "%REPO_ROOT%"
    exit /b 1
)

for %%M in (%MODELS%) do (
    echo [%DATE% %TIME%] model running - %%M
    set "CMD=%PYTHON% -m scripts.base_eval --eval %EVAL% --hf-path %%M --max-per-task %MAX_PER_TASK%"
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
    )
)

popd >nul
exit /b %OVERALL_RC%

:usage_ok
echo Usage:
echo   runs\base_eval_hf_benchmark.bat [options]
echo.
echo Options:
echo   --python ^<exe^>              Python executable. Default: python
echo   --eval ^<modes^>              Eval mode(s) for scripts.base_eval. Default: core
echo   --max-per-task ^<n^>          Max examples per CORE task. Default: -1
echo   --device-type ^<type^>        cuda^|cpu^|mps ^(empty = autodetect^)
echo   --nanochat-base-dir ^<path^>  Sets NANOCHAT_BASE_DIR for this run
echo   --models "^<m1 m2 ...^>"      Space- or comma-separated HF model list
echo   --stop-on-error               Stop on first failing model
echo   --dry-run                     Print commands without executing
echo   -h, --help                    Show this help
exit /b 0

:usage_err
call :usage_ok
exit /b 1
