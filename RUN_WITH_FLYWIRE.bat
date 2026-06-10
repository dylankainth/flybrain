@echo off
REM Run simulator with downloaded FlyWire data

echo ============================================================
echo FRUIT FLY BRAIN DRONE - FlyWire FAFB v783 Edition
echo ============================================================
echo.

REM Activate venv
call venv\Scripts\activate.bat

REM Parse FlyWire data
echo [1/2] Parsing FlyWire data files...
python parse_flywire.py

if errorlevel 1 (
    echo [!] Parsing failed. Check your downloaded files.
    pause
    exit /b 1
)

echo.
echo [2/2] Running simulator with REAL FLY BRAIN...
echo.

python simulate_headless.py

pause
