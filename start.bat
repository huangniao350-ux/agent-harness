@echo off
title AgentHarness - Quick Start
cd /d "%~dp0"

echo ================================================
echo   AgentHarness 快速启动
echo ================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)

python -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
    echo [提示] 缺少 API 依赖，正在安装 fastapi / uvicorn ...
    pip install "fastapi>=0.110" "uvicorn>=0.29" --quiet
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8765 ^| findstr LISTENING') do (
    echo [提示] 停止占用 8765 端口的旧实例 (PID %%a)
    taskkill /F /PID %%a >nul 2>&1
)

echo [启动] 服务启动中，浏览器将自动打开...
echo.
echo   开发者控制台  http://127.0.0.1:8765
echo   API 文档      http://127.0.0.1:8765/docs
echo.
echo   停止服务：在本窗口按 Ctrl+C，或双击 stop.bat
echo ================================================

start "" cmd /c "ping -n 6 127.0.0.1 >nul & start "" http://127.0.0.1:8765"

python -m agent_harness server --port 8765
pause
