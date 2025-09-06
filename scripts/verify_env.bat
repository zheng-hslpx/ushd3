@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM === 基于脚本路径定位 venv ===
set "SCRIPT_DIR=%~dp0"
set "VENV=%SCRIPT_DIR%..\venv"

if not exist "%VENV%\Scripts\python.exe" (
  echo [ERROR] 未找到虚拟环境：%VENV%
  echo 请先运行 scripts\setup_env.bat
  pause
  exit /b 1
)

echo [*] Activating venv...
call "%VENV%\Scripts\activate"
if errorlevel 1 (
  echo [ERROR] 激活虚拟环境失败
  pause
  exit /b 1
)

echo [*] 导入并打印关键包版本（忽略 Gym 提示属正常）...
"%VENV%\Scripts\python.exe" - <<PY
import sys
try:
    import torch, gym, numpy, matplotlib
    try:
        import visdom
        has_visdom = True
    except Exception:
        has_visdom = False
    print("OK ✓",
          "python", sys.version.split()[0],
          "torch", getattr(torch, "__version__", "?"),
          "gym", getattr(gym, "__version__", "?"),
          "numpy", getattr(numpy, "__version__", "?"),
          "matplotlib", getattr(matplotlib, "__version__", "?"),
          "visdom", (visdom.__version__ if has_visdom else "not_installed"))
except Exception as e:
    print("ERROR:", e)
    raise
PY

if errorlevel 1 (
  echo [ERROR] 有包导入失败，请查看上面的报错
  pause
  exit /b 1
)

echo [OK] 环境检查通过
pause
endlocal
