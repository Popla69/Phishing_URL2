@echo off
cls
echo.
echo 🛡️  PHISHING DETECTOR WEB SERVER
echo =====================================
echo.
echo Starting web server...
echo.
echo 🌐 Web Interface will be available at:
echo    http://localhost:8000
echo.
echo 💡 Features:
echo    ✅ Single URL checking
echo    ✅ Batch URL processing  
echo    ✅ File upload support
echo    ✅ Real-time results
echo.
echo 🛑 Press Ctrl+C to stop the server
echo =====================================
echo.

python web_server.py

echo.
echo Server stopped.
pause
