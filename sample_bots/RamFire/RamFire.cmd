cd /d "%~dp0"

call ..\deps\install-dependencies.cmd

set "VENV_PY=..\deps\venv\Scripts\python.exe"
if exist "%VENV_PY%" (
  "%VENV_PY%" "RamFire.py"
  exit /b %errorlevel%
)

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python3 >nul 2>nul && set "PY=python3"
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Error: Python not found. Please install Python 3 and ensure either 'py' or 'python' is on PATH.
  exit /b 1
)
%PY% "RamFire.py"