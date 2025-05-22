@echo off
REM Run the LightningImageDiffusion training script
setlocal
set PYTHONIOENCODING=utf-8

REM Activate your Python environment if needed
REM call path\to\venv\Scripts\activate

python train.py

pause
