@echo off
setlocal EnableDelayedExpansion

echo ============================
echo    AI Trash Launcher
echo ============================
echo.

set "HAS_GPU=0"
set "HAS_CPU=0"

if exist "venv_gpu\Scripts\python.exe" set "HAS_GPU=1"
if exist "venv_cpu\Scripts\python.exe" set "HAS_CPU=1"

if %HAS_GPU%==1 if %HAS_CPU%==1 goto :both_found
if %HAS_GPU%==1 goto :gpu_only
if %HAS_CPU%==1 goto :cpu_only

echo No venv found. Run setup.bat first.
pause
exit /b 1

:both_found
echo Both environments found.
echo   [1] CPU
echo   [2] GPU (CUDA)
echo.
set /p CHOICE="Select: "
if "%CHOICE%"=="2" (
    call venv_gpu\Scripts\activate.bat
) else (
    call venv_cpu\Scripts\activate.bat
)
goto :run

:gpu_only
echo Found GPU environment.
call venv_gpu\Scripts\activate.bat
goto :run

:cpu_only
echo Found CPU environment.
call venv_cpu\Scripts\activate.bat
goto :run

:run
start /wait python app.py