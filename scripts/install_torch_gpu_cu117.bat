@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM === 定位 venv / python / pip ===
set "SCRIPT_DIR=%~dp0"
set "ROOT_DIR=%SCRIPT_DIR%.."
set "VENV=%ROOT_DIR%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "PIP=%VENV%\Scripts\pip.exe"

if not exist "%PY%" (
  echo [ERROR] 未找到虚拟环境：%VENV%
  echo 请先运行 scripts\setup_env.bat
  pause & exit /b 1
)

echo [*] 激活虚拟环境...
call "%VENV%\Scripts\activate" || (echo [ERROR] 激活失败 & pause & exit /b 1)

echo [*] 卸载现有 torch/torchvision/torchaudio（若已安装）...
"%PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul

echo [*] 清理 pip 缓存...
"%PY%" -m pip cache purge >nul 2>nul

REM ===== 安装 CUDA 11.7 版 =====
set "INDEX_CU117=https://download.pytorch.org/whl/cu117"
echo [*] 安装 PyTorch (CUDA 11.7) ...
"%PIP%" install --index-url %INDEX_CU117% torch torchvision

if errorlevel 1 (
  echo [ERROR] CUDA 11.7 安装失败，请检查网络或稍后重试。
  pause & exit /b 1
)

REM 验证 CUDA 可用性
for /f %%i in ('"%PY%" -c "import torch;print(torch.cuda.is_available())"') do set "CUDAOK=%%i"
for /f "delims=" %%j in ('"%PY%" -c "import torch,sys;print(getattr(torch.version,'cuda',None) or '')"') do set "CUDAVER=%%j"

echo [*] 版本信息：
"%PY%" -c "import torch;print('torch',torch.__version__,'cuda?',torch.cuda.is_available(),'cu',getattr(torch.version,'cuda',None))"
"%PY%" -c "import torchvision;print('torchvision',torchvision.__version__)"

if /I "%CUDAOK%"=="True" (
  echo [OK] 已安装 CUDA 11.7 版 PyTorch（cu=%CUDAVER%），CUDA 可用：%CUDAOK%
) else (
  echo [WARN] CUDA 不可用，请确认显卡驱动是否支持 CUDA 11.7 并已正确安装。
)

echo.
echo 之后使用： scripts\run_train.bat --config config.json
pause
endlocal
