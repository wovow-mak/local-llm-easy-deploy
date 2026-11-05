@echo off
title Local LLM - deploy.py
cd /d "%~dp0"

rem %* 表示"把用户传进来的参数原样转交给 deploy.py"
rem 例如: deploy.bat --check   或   deploy.bat --cpu
C:\Python312\python.exe "%~dp0deploy.py" %*

echo.
echo ==================================================
echo  Program exited. Press any key to close window.
echo ==================================================
pause >nul
