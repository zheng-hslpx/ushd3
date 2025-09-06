@echo off
setlocal ENABLEDELAYEDEXPANSION
chcp 65001 >nul
set PYTHONUTF8=1

REM ===== Settings =====
set VENV_DIR=venv
set REQ_FILE=requirements.txt
set LOCAL_CACHE=%CD%\pip-cache
set INDEX_URL=https://mirrors.aliyun.com/pypi/simple/

echo [1/7] Checking Python...
where python >nul 2>nul && (set PY_EXE=python) || (where py >nul 2>nul && (set PY_EXE=py) || (echo [ERROR] 未找到 Python。请安装后再试& pause & exit /b 1))

if not exist "%REQ_FILE%" (
    echo [ERROR] 未找到 %REQ_FILE% （当前目录：%cd%）
    pause & exit /b 1
)

echo [2/7] Creating virtual environment (if missing)...
if not exist "%VENV_DIR%\Scripts\python.exe" (
    %PY_EXE% -m venv "%VENV_DIR%" || (echo [ERROR] 创建虚拟环境失败& pause & exit /b 1)
) else (
    echo      已检测到现有虚拟环境：%VENV_DIR%
)

echo [3/7] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate" || (echo [ERROR] 激活虚拟环境失败& pause & exit /b 1)

echo [4/7] 准备本地 pip 缓存目录（避免寫入用戶全局緩存）...
if exist "%LOCAL_CACHE%" (echo      使用已有：%LOCAL_CACHE%) else (mkdir "%LOCAL_CACHE%")
set PIP_CACHE_DIR=%LOCAL_CACHE%

REM --- 关键步骤：先无缓存安装 gym，彻底绕开全局缓存 ---
echo [5/7] Pre-install gym (Windows 条件包) with --no-cache-dir ...
pip install --no-cache-dir --index-url "%INDEX_URL%" "gym>=0.26,<0.27"
if not %ERRORLEVEL%==0 (
    echo [WARN] gym 无缓存安装失败，改走兩步回退：
    echo        1) 切到本地缓存  2) 仍不行再用無缓存最終重試
    set PIP_CACHE_DIR=%LOCAL_CACHE%
    pip install --index-url "%INDEX_URL%" "gym>=0.26,<0.27" || (
        set PIP_CACHE_DIR=
        pip install --no-cache-dir --index-url "%INDEX_URL%" "gym>=0.26,<0.27" || (
            echo [ERROR] gym 安装仍失败，請檢查殺毒/權限或執行 ^"pip cache purge^" 後重試
            pause & exit /b 1
        )
    )
)

echo [6/7] 安装剩余依赖（requirements.txt）...
REM 先常规 -> 失败用本地缓存 -> 仍失败最后無缓存
pip install --index-url "%INDEX_URL%" -r "%REQ_FILE%" && goto OK_INSTALL
echo [WARN] 常规安装失败，使用本地缓存重试...
set PIP_CACHE_DIR=%LOCAL_CACHE%
pip install --index-url "%INDEX_URL%" -r "%REQ_FILE%" && goto OK_INSTALL
echo [WARN] 本地缓存仍失败，改用無缓存最終重試...
set PIP_CACHE_DIR=
pip install --no-cache-dir --index-url "%INDEX_URL%" -r "%REQ_FILE%" || (
    echo [ERROR] 依赖安装失败。可嘗試：
    echo         1) 在已激活 venv 內執行：pip cache purge
    echo         2) 臨時關閉殺毒或將 C:\Users\lenovo\AppData\Local\pip\cache 加白
    echo         3) 重跑本腳本
    pause & exit /b 1
)

:OK_INSTALL
echo [7/7] Writing VSCode settings to .vscode\settings.json ...
if not exist ".vscode" mkdir ".vscode"
(
echo {
echo     "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
echo     "python.terminal.activateEnvironment": true
echo }
) > ".vscode\settings.json"

echo.
echo ==========================================
echo  完成 ✅
echo  - 虚拟环境：%VENV_DIR%
echo  - VSCode 将自动使用该解释器
echo  - 手动激活：%VENV_DIR%\Scripts\activate
echo ==========================================
echo.
pause
endlocal
