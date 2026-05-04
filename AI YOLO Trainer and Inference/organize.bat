@echo off
setlocal

echo ============================
echo    Dataset Organizer
echo ============================
echo.

if exist "venv_gpu\Scripts\python.exe" (
    call venv_gpu\Scripts\activate.bat
    goto :run
)

if exist "venv_cpu\Scripts\python.exe" (
    call venv_cpu\Scripts\activate.bat
    goto :run
)

echo No venv found. Run setup.bat first.
pause
exit /b 1

:run
python organize_dataset.py
pause