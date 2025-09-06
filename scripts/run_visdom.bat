@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM 用脚本目录定位 venv
set "SCRIPT_DIR=%~dp0"
set "VENV=%SCRIPT_DIR%..\venv"
set "PY=%VENV%\Scripts\python.exe"

REM 参数1=端口(默认8097)，参数2=模式(fg/bg，默认fg)，参数3=host(默认localhost)
set "PORT=8097"
if not "%~1"=="" set "PORT=%~1"
set "MODE=fg"
if /I "%~2"=="bg" set "MODE=bg"
set "HOST=localhost"
if not "%~3"=="" set "HOST=%~3"

if not exist "%PY%" (
  echo [ERROR] 未找到虚拟环境：%VENV%
  echo 请先运行 scripts\setup_env.bat
  pause & exit /b 1
)

echo [*] 检查端口占用: %PORT%
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":%PORT% " ^| findstr LISTENING') do (
  echo [WARN] 端口 %PORT% 被占用，PID=%%p
  echo        如需释放： taskkill /PID %%p /F
)

echo [*] 激活虚拟环境...
call "%VENV%\Scripts\activate" || (echo [ERROR] 激活失败 & pause & exit /b 1)

if /I "%MODE%"=="bg" (
  echo [*] 后台启动 Visdom（看不到日志）。端口: %PORT% host: %HOST%
  start "" "%PY%" -m visdom.server -port %PORT% --hostname %HOST%
  echo [OK] 已尝试后台启动。稍后访问: http://%HOST%:%PORT%
  pause & exit /b 0
) else (
  echo [*] 前台启动 Visdom（可查看日志）。端口: %PORT% host: %HOST%
  REM 先尝试使用单横杠 -logging_level（兼容多数版本）
  "%PY%" -m visdom.server -port %PORT% --hostname %HOST% -logging_level DEBUG
  if errorlevel 1 (
    echo [WARN] -logging_level 参数不可用，改为不带日志级别重试...
    "%PY%" -m visdom.server -port %PORT% --hostname %HOST%
  )
  echo [*] 进程已退出（如异常请上滚查看日志）
  pause & exit /b 0
)
