@echo off
chcp 65001 >nul
:: ==================================================
:: Robust training script with absolute path handling
:: ==================================================

:: [1] Get project root (assuming .bat is in scripts/)
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"

:: [2] Set default config path
set "DEFAULT_CONFIG=%PROJECT_ROOT%\config\improved_config.json"

:: [3] Allow custom config via parameter
if not "%~1"=="" (
    set "CONFIG=%~1"
    :: Convert relative to absolute path if needed
    if not "%~f1"=="%~1" (
        set "CONFIG=%PROJECT_ROOT%\%~1"
    )
) else (
    set "CONFIG=%DEFAULT_CONFIG%"
)

:: [4] Validate config file
if not exist "%CONFIG%" (
    echo [ERROR] Config file not found: %CONFIG%
    echo Valid examples:
    echo   .\scripts\run_train.bat config\custom.json
    echo   .\scripts\run_train.bat E:\full\path\to\config.json
    pause
    exit /b 1
)

echo [*] Using config: %CONFIG%

:: [5] Activate venv
echo [*] Activating virtual environment...
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    call "%PROJECT_ROOT%\venv\Scripts\activate.bat"
) else (
    echo [ERROR] venv not found. Run scripts\setup_env.bat first.
    pause
    exit /b 1
)

:: [6] Start training
echo [*] Starting training...
python "%PROJECT_ROOT%\train.py" --config "%CONFIG%"
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Training failed (exit code: %ERRORLEVEL%)
    pause
    exit /b %ERRORLEVEL%
)

:: [7] Show save directory info
for /f "usebackq tokens=*" %%i in (`
  powershell -Command "(gc '%CONFIG%' | ConvertFrom-Json).train_paras.viz_name"
`) do set "VIZNAME=%%~i"

echo ============================================
echo [INFO] Visdom Environment: "%VIZNAME%"
echo Access: http://localhost:8097
echo ============================================

pause