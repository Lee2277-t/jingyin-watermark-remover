@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title 净印 · 批量去水印
cd /d "%~dp0"

set "PYW=%~dp0.venv\Scripts\pythonw.exe"
if not exist "%PYW%" (
  echo [启动失败] 找不到运行环境 .venv\Scripts\pythonw.exe
  echo 请先在工具目录里执行：py -3.11 -m venv .venv  然后  .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

start "" /b "%PYW%" server.py

echo 正在确认本地服务是否真的起来了...

rem 真探活：读 work\runtime.json 里的端口，问 /api/health 是不是本工具在应答，最多等 12 秒
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $url=''; for($i=0;$i -lt 24;$i++){ try{ $r=Get-Content -Raw -Encoding UTF8 'work\runtime.json'|ConvertFrom-Json; $h=Invoke-RestMethod -UseBasicParsing -TimeoutSec 3 ('http://127.0.0.1:'+$r.port+'/api/health'); if($h.app -eq 'jingyin' -and $h.instance -eq $r.instance){ $url='http://127.0.0.1:'+$r.port+'/'; break } } catch{}; Start-Sleep -Milliseconds 500 }; if($url){ Write-Output $url; exit 0 }; exit 1"
if errorlevel 1 (
  echo.
  echo [启动失败] 本地服务 12 秒内没有应答，浏览器里没有打开工具。
  echo 常见原因有两个：端口 8765-8785 全被别的程序占用；或者 .venv 运行环境坏了。
  echo 请打开 work\tool.log 查看错误。分享日志前，请遮住本机路径和文件名。
  pause
  exit /b 1
)

echo 本地服务已就绪，上面这行就是本机地址，这个窗口可以关闭。
ping -n 3 127.0.0.1 >nul
exit /b 0
