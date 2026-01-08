@echo off
REM Script to run a command 5 times and calculate test AUC statistics
REM Usage: run_5times.bat <command>
REM Example: run_5times.bat "python train_dann.py --dataset goodhiv_scaffold_covariate --epochs 10"

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Error: No command provided
    echo Usage: run_5times.bat "your command here"
    echo Example: run_5times.bat "python train_dann.py --dataset goodhiv_scaffold_covariate"
    exit /b 1
)

REM Redirect to Python script for better statistics handling
python run_multiple.py 5 %*

endlocal
