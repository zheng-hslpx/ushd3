@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM === 定位到 venv 和 python ===
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

echo [*] GPU/驱动信息（若存在 nvidia-smi 会显示）...
where nvidia-smi >nul 2>nul && (nvidia-smi) || (echo    未检测到 nvidia-smi（无驱动或未在 PATH 中，若后续 CUDA 不可用请先更新顯卡驅動）)

echo [*] 卸载现有 torch/torchvision/torchaudio（若已安装）...
"%PY%" -m pip uninstall -y torch torchvision torchaudio >nul 2>nul

echo [*] 清理 pip 缓存...
"%PY%" -m pip cache purge >nul 2>nul

REM === 尝试安装 CUDA 12.1 ===
set "TORCH_INDEX_CU121=https://download.pytorch.org/whl/cu121"
echo [1/2] 安装 PyTorch (CUDA 12.1) ...
"%PIP%" install --index-url %TORCH_INDEX_CU121% torch torchvision
if errorlevel 1 (
  echo [WARN] CUDA 12.1 安装过程报错，准备回退到 CUDA 11.8...
  goto INSTALL_CU118
)

REM 验证 CUDA 可用性（True/False）
for /f %%i in ('"%PY%" -c "import torch;print(torch.cuda.is_available())"') do set "CUDAOK=%%i"
for /f "delims=" %%j in ('"%PY%" -c "import torch,sys;print(getattr(torch.version,'cuda',None) or '')"') do set "CUDAVER=%%j"
if /I "%CUDAOK%"=="True" (
  echo [OK] 已安装 CUDA 版 PyTorch（cu=%CUDAVER%），CUDA 可用：%CUDAOK%
  goto SHOW_VERS
) else (
  echo [WARN] 安装完成但 CUDA 不可用（可能驱动不匹配/过旧），回退到 CUDA 11.8 再试...
  goto INSTALL_CU118
)

:INSTALL_CU118
echo [2/2] 安装 PyTorch (CUDA 11.8) ...
"%PIP%" install --index-url https://download.pytorch.org/whl/cu118 torch torchvision
if errorlevel 1 (
  echo [ERROR] CUDA 11.8 安装也失败。请检查网络或稍后重试。
  echo        - 可尝试：更换网络/代理；更新显卡驱动；或先用 CPU 版临时运行
  pause & exit /b 1
)

for /f %%i in ('"%PY%" -c "import torch;print(torch.cuda.is_available())"') do set "CUDAOK=%%i"
for /f "delims=" %%j in ('"%PY%" -c "import torch,sys;print(getattr(torch.version,'cuda',None) or '')"') do set "CUDAVER=%%j"
if /I "%CUDAOK%"=="True" (
  echo [OK] 已安装 CUDA 版 PyTorch（cu=%CUDAVER%），CUDA 可用：%CUDAOK%
) else (
  echo [WARN] 安装完成但 CUDA 仍不可用：%CUDAOK%
  echo       常见原因：顯卡驅動過舊或未安裝；無獨顯；遠程桌面限制等
)

:SHOW_VERS
echo [*] 版本信息：
"%PY%" - <<PY
import torch, torchvision
print("torch:", torch.__version__,
      "cuda_available:", torch.cuda.is_available(),
      "cuda_runtime:", getattr(torch.version, "cuda", None))
print("torchvision:", torchvision.__version__)
PY

echo.
echo 提示：
echo - 若 CUDA 不可用，請更新 NVIDIA 顯卡驅動到對應版本，再重跑本腳本。
echo - 之後用 scripts\run_train.bat 啟動訓練，會自動使用 GPU（若可用）。
pause
endlocal
