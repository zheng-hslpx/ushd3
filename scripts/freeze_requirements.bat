@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM 自动用脚本自身目录定位 venv
set "SCRIPT_DIR=%~dp0"
set "VENV=%SCRIPT_DIR%..\venv"
set "LOCK_FILE=%SCRIPT_DIR%..\requirements.lock.txt"

if not exist "%VENV%\Scripts\python.exe" (
  echo [ERROR] 未找到虚拟环境：%VENV%
  echo 请先运行 scripts\setup_env.bat
  pause
  exit /b 1
)

echo [*] Activating venv...
call "%VENV%\Scripts\activate"

echo [*] 生成锁定依赖到 %LOCK_FILE% ...
"%VENV%\Scripts\python.exe" -m pip freeze > "%LOCK_FILE%"

if %ERRORLEVEL%==0 (
  echo [OK] 已生成依赖锁定文件：%LOCK_FILE%
) else (
  echo [ERROR] 生成失败，请检查 pip freeze 输出
  pause
  exit /b 1
)
pause
endlocal
