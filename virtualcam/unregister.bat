@echo off
echo ============================================
echo  Kagerou Virtual Camera - Unregistration
echo  Requires Administrator privileges
echo ============================================

net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"%~f0\"' -Verb RunAs"
    exit /b
)

echo Unregistering...
regsvr32 /u /s "%~dp0..\bin\kagerou_virtualcam.dll"
if %ERRORLEVEL% EQU 0 (
    echo SUCCESS: Kagerou Virtual Camera unregistered.
) else (
    echo Registry cleanup may have partially failed.
)
pause
