@echo off
chcp 65001 >nul
echo ========================================
echo   安装每日任务开机自启动
echo ========================================
echo.

set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fi"
set "STARTUP_SCRIPT=%PROJECT_ROOT%\scripts\start_daily_task.bat"
set "STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP_FOLDER%\GluttonousDailyTask.lnk"

echo 项目目录: %PROJECT_ROOT%
echo 启动脚本: %STARTUP_SCRIPT%
echo 启动文件夹: %STARTUP_FOLDER%
echo.

:: 使用 PowerShell 创建快捷方式
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%STARTUP_SCRIPT%'; $s.WorkingDirectory = '%PROJECT_ROOT%'; $s.WindowStyle = 7; $s.Description = 'Gluttonous Daily Task'; $s.Save()"

if exist "%SHORTCUT%" (
    echo [成功] 已创建开机自启动快捷方式
    echo.
    echo 快捷方式位置: %SHORTCUT%
    echo.
    echo 系统重启后，每日任务将自动在后台启动。
) else (
    echo [失败] 创建快捷方式失败
)

echo.
pause
