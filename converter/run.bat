@echo off
rem Opens a command prompt in which dual-convert works from any folder.
rem The folder's path goes through a variable read after parsing (!ACTIVATE!), so
rem characters such as & ( ) %% and ! in it are never parsed as commands.
setlocal DisableDelayedExpansion
set "ACTIVATE=%~dp0venv\Scripts\activate.bat"
if not exist "%ACTIVATE%" (
    echo dual-convert is not installed yet: run "install venv.bat" first.
    pause
    exit /b 1
)
setlocal EnableDelayedExpansion
cmd /k call "!ACTIVATE!" ^&^& echo. ^&^& echo dual-convert is ready. For example: ^&^& echo   dual-convert model.safetensors --size 960x540 ^&^& echo   dual-convert model.pth --min 256x256 --opt 1280x720 --max 1920x1088 ^&^& echo All options: dual-convert -help ^&^& echo.
