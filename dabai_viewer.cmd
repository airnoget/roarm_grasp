@echo off
setlocal
pushd "%~dp0.orbbec-deps\viewer-v1.10.37"
if errorlevel 1 exit /b 1
"OrbbecViewer.exe"
set "viewer_exit_code=%ERRORLEVEL%"
popd
exit /b %viewer_exit_code%
