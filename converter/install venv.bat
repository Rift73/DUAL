@echo off
rem Creates venv\ beside this file with the packages dual-convert needs, for one
rem TensorRT version: "install venv.bat" [11.4.0.106 | 11.3.0.99 | 11.2.1.2]
rem Without an argument it asks. Running it again with another version switches
rem the TensorRT packages.
setlocal EnableExtensions DisableDelayedExpansion
set "HERE=%~dp0"
set "VENV=%HERE%venv"
set "TRT=%~1"
rem Started without an argument (double-clicked): keep the window open at the end.
set "ASKED="
if "%TRT%"=="" set "ASKED=1"
if not "%TRT%"=="" goto check_version
echo Which TensorRT version should dual-convert use?
echo   1  11.4.0.106  (default)
echo   2  11.3.0.99
echo   3  11.2.1.2
echo An engine only loads in the TensorRT version that built it.
set "CHOICE="
set /p "CHOICE=Type 1, 2 or 3, or press Enter for 11.4.0.106: "
rem A typed quote would break the comparisons below: it becomes a mark that matches no
rem choice, so an answer with a quote is refused.
if defined CHOICE set "CHOICE=%CHOICE:"=(quote)%"
if "%CHOICE%"=="" set "TRT=11.4.0.106"
if "%CHOICE%"=="1" set "TRT=11.4.0.106"
if "%CHOICE%"=="2" set "TRT=11.3.0.99"
if "%CHOICE%"=="3" set "TRT=11.2.1.2"
if "%TRT%"=="" (
    echo "%CHOICE%" is not one of the choices.
    goto failed_quietly
)

:check_version
echo TensorRT version: %TRT%
if "%TRT%"=="11.4.0.106" goto find_python
if "%TRT%"=="11.3.0.99" goto find_python
if "%TRT%"=="11.2.1.2" goto find_python
echo TensorRT %TRT% is not offered: choose 11.4.0.106, 11.3.0.99 or 11.2.1.2.
goto failed_quietly

:find_python
if not exist "%VENV%\Scripts\python.exe" goto new_venv
"%VENV%\Scripts\python.exe" -c "pass" >nul 2>&1 && goto install
echo The existing venv folder does not work (its Python does not start, for example
echo because that Python was removed or moved). Delete the folder
echo   %VENV%
echo and run this file again.
goto failed_quietly

:new_venv
set "PYTHON="
for %%V in (3.14 3.13 3.12) do (
    if not defined PYTHON (
        py -%%V -c "import sys; sys.exit(sys.maxsize < 2**32)" >nul 2>&1 && set "PYTHON=py -%%V"
    )
)
if not defined PYTHON (
    python -c "import sys; sys.exit(not ((3, 12) <= sys.version_info[:2] <= (3, 14) and sys.maxsize > 2**32))" >nul 2>&1 && set "PYTHON=python"
)
if not defined PYTHON (
    echo No usable Python was found. Install 64-bit Python 3.14, 3.13 or 3.12 from
    echo https://www.python.org/downloads/windows/ and run this file again.
    goto failed_quietly
)
echo Creating venv with %PYTHON% ...
%PYTHON% -m venv "%VENV%"
if errorlevel 1 (
    echo Creating the venv failed.
    goto failed_quietly
)

:install
set "PY=%VENV%\Scripts\python.exe"
echo Installing PyTorch (CPU), ONNX, Triton and the other packages ...
"%PY%" -m pip install --disable-pip-version-check --extra-index-url https://download.pytorch.org/whl/cpu -r "%HERE%requirements.txt"
if errorlevel 1 goto failed
echo Installing TensorRT %TRT% ...
rem Another version's three packages go first, so that pip never sees the meta package
rem of one version next to the libraries of another.
"%PY%" -c "import sys; from importlib.metadata import version; sys.exit(version('tensorrt_cu13_libs') != sys.argv[1])" %TRT% >nul 2>&1
if errorlevel 1 "%PY%" -m pip uninstall --disable-pip-version-check -y -q tensorrt-cu13 tensorrt_cu13_bindings tensorrt_cu13_libs
"%PY%" -m pip install --disable-pip-version-check --extra-index-url https://pypi.nvidia.com "tensorrt_cu13_libs==%TRT%" "tensorrt_cu13_bindings==%TRT%"
if errorlevel 1 goto failed
"%PY%" -m pip install --disable-pip-version-check --no-deps "tensorrt-cu13==%TRT%"
if errorlevel 1 goto failed
rem dual-convert runs from this folder: a .pth file puts it on the venv's import path
rem and dual-convert.cmd in venv\Scripts starts it.
"%PY%" -c "import pathlib, sys, sysconfig; home = pathlib.Path(sys.argv[1]).resolve(); pathlib.Path(sysconfig.get_paths()['purelib'], 'dual_convert.pth').write_text(str(home) + '\n', encoding='utf-8'); pathlib.Path(sys.executable).with_name('dual-convert.cmd').write_text('@\"%%~dp0python.exe\" -m dual_convert %%*\r\n', encoding='ascii', newline='')" "%HERE%."
if errorlevel 1 goto failed
echo Checking the installation ...
"%PY%" -m dual_convert.environment
if errorlevel 1 goto failed
echo.
echo Done. Start run.bat to use dual-convert.
set "CODE=0"
goto finish

:failed
echo.
echo The installation failed; the messages above say why.

:failed_quietly
set "CODE=1"

:finish
if defined ASKED pause
exit /b %CODE%
