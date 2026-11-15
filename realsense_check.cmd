@echo off
setlocal
cd /d "%~dp0"
if not exist "depth_diagnostics" mkdir "depth_diagnostics"
echo RealSense depth-only check. Close other camera apps first.
echo Log: %CD%\depth_diagnostics\realsense-startup.log
"D:\anaconda\python.exe" -X faulthandler -u "depth_health.py" --camera realsense --depth-only --rotate 180 > "depth_diagnostics\realsense-startup.log" 2>&1
set "rs_exit_code=%ERRORLEVEL%"
type "depth_diagnostics\realsense-startup.log"
echo.
echo Exit code: %rs_exit_code%
pause
exit /b %rs_exit_code%
