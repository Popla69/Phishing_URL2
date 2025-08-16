@echo off
echo.
echo 🛡️  PHISHING URL DETECTOR
echo ========================
echo.

if "%1"=="" (
    echo Usage: check_phishing.bat [URL]
    echo.
    echo Examples:
    echo   check_phishing.bat google.com
    echo   check_phishing.bat suspicious-site.com
    echo   check_phishing.bat "http://phishing-example.com"
    echo.
    set /p url="Enter URL to check: "
    if not "!url!"=="" (
        python check_url.py "!url!"
    )
) else (
    python check_url.py "%1"
)

echo.
pause
