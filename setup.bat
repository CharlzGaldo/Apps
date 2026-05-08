@echo off
setlocal EnableDelayedExpansion

echo ========================================
echo   AI Trash - Environment Setup
echo ========================================
echo.

:: Check for CUDA
set "HAS_CUDA=0"
where nvidia-smi >nul 2>&1 && set "HAS_CUDA=1"

if %HAS_CUDA%==0 (
    echo [INFO] No CUDA detected on your system.
    echo [INFO] You can only use CPU mode.
    echo.
    set "MODE=cpu"
    goto :create_env
)

echo Your system has CUDA available.
echo.
echo Select PyTorch version:
echo   [1] CPU only
echo   [2] GPU (CUDA)
echo.
set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="2" (
    set "MODE=gpu"
) else (
    set "MODE=cpu"
)

:create_env
set "VENV_NAME=venv_%MODE%"
echo.
echo Creating %VENV_NAME%...
echo.

:: Find Python 3.12
set "PYTHON_CMD="
for %%P in (python3.12 python py) do (
    for /f "tokens=*" %%V in ('%%P --version 2^>nul') do (
        echo %%V | findstr "3.12" >nul && set "PYTHON_CMD=%%P" && goto :found
    )
)

echo ERROR: Python 3.12 not found in PATH.
echo Please install Python 3.12.10 or update PATH.
pause
exit /b 1

:found
echo Using: %PYTHON_CMD%
%PYTHON_CMD% -m venv %VENV_NAME%
call %VENV_NAME%\Scripts\activate.bat

python -m pip install --upgrade pip

if "%MODE%"=="gpu" (
    pip install -r requirements_gpu.txt
) else (
    pip install -r requirements_cpu.txt
)

echo.
echo ========================================
echo   Done! Environment: %VENV_NAME%
echo   Activate with: %VENV_NAME%\Scripts\activate.bat
echo ========================================
pause