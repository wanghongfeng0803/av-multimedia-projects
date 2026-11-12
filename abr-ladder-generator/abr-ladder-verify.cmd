@echo off
setlocal
python "%~dp0abr-ladder-verify.py" %*
exit /b %ERRORLEVEL%
