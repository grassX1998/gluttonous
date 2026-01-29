@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ========================================
echo   Gluttonous 每日任务启动脚本
echo ========================================
echo.

:: 获取脚本所在目录的父目录（项目根目录）
set "SCRIPT_DIR=%~dp0"
for %%i in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fi"

set "LOCK_FILE=%PROJECT_ROOT%\.pipeline_data\daily_task.lock"
set "STATUS_FILE=%PROJECT_ROOT%\DATA_COLLECTION_STATUS.json"

cd /d "%PROJECT_ROOT%"
echo [1/4] 项目目录: %PROJECT_ROOT%

:: 检查并清理旧进程
echo [2/4] 检查现有进程...

if exist "%LOCK_FILE%" (
    set /p OLD_PID=<"%LOCK_FILE%"
    echo       发现锁文件，PID: !OLD_PID!

    :: 检查进程是否存在
    tasklist /FI "PID eq !OLD_PID!" 2>nul | find "!OLD_PID!" >nul
    if !errorlevel! equ 0 (
        echo       进程正在运行，正在终止...
        taskkill /PID !OLD_PID! /F >nul 2>&1
        timeout /t 1 /nobreak >nul
        echo       已终止旧进程
    ) else (
        echo       旧进程已不存在
    )

    del /f "%LOCK_FILE%" >nul 2>&1
    echo       已删除锁文件
) else (
    echo       无现有进程
)

:: 启动新进程
echo [3/4] 启动每日任务...

if "%1"=="--test" (
    echo       模式: 测试（立即执行一次）
    echo.
    python -m pipeline.data_collection.daily_task --test
    goto :end
)

echo       模式: 后台调度（17:00 后自动采集）

:: 使用 PowerShell 在后台启动并获取 PID
for /f "tokens=*" %%i in ('powershell -Command "Start-Process -FilePath 'python' -ArgumentList '-m', 'pipeline.data_collection.daily_task' -WindowStyle Hidden -PassThru | Select-Object -ExpandProperty Id"') do set NEW_PID=%%i

echo       新进程 PID: %NEW_PID%

:: 等待状态文件更新
timeout /t 3 /nobreak >nul

echo [4/4] 检查运行状态...

if exist "%STATUS_FILE%" (
    echo.
    echo ========================================
    echo   任务状态
    echo ========================================
    type "%STATUS_FILE%"
    echo.
    echo ========================================
)

echo.
echo 每日任务已在后台启动！
echo.

:: 获取正确格式的日期
for /f "tokens=*" %%d in ('powershell -Command "Get-Date -Format 'yyyy-MM-dd'"') do set TODAY=%%d

echo 常用命令:
echo   查看状态:  type DATA_COLLECTION_STATUS.json
echo   查看日志:  type logs\%TODAY%\data_collection.log
echo   停止任务:  taskkill /PID %NEW_PID% /F

:end
endlocal
