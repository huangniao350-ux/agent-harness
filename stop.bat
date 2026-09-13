@echo off
title AgentHarness - Stop

set FOUND=0
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    set FOUND=1
)
if "%FOUND%"=="1" (
    echo [完成] AgentHarness 服务已停止
) else (
    echo [提示] 8765 端口没有正在运行的服务
)
ping -n 3 127.0.0.1 >nul
