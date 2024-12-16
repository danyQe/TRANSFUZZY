@echo off
REM Create a virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate

@REM REM UPGRADING pip
@REM python -m pip install --upgrade pip

REM Install dependencies from requirements.txt
pip install -r requirements.txt

echo.
echo Virtual environment setup complete!
echo.

REM Ask user if they want to run main.py
set /p RUN_MAIN="Do you want to run main.py? (Y/N): "

IF /I "%RUN_MAIN%"=="Y" (
    REM Run main.py
    python main.py

    REM Open localhost:5000 in the default browser
    start http://localhost:5000
) ELSE (
    echo Skipping main.py execution.
)

pause
cmd /k
