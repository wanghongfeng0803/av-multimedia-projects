@echo off
setlocal
python "%~dp0abr-ladder.py" %*
exit /b %ERRORLEVEL%
