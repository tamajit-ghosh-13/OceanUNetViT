@echo off
echo ==============================================================================
echo  LAUNCHING OCEANEMBED AI INTELLIGENCE PLATFORM (DUO-ELITE v5.0)
echo ==============================================================================

cd /d "%~dp0"

echo [1/2] Starting Python FastAPI Inference Server on Port 8000...
start "OceanEmbed API Server (Port 8000)" cmd /c "python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --log-level info"

timeout /t 3 /nobreak >nul

echo [2/2] Starting Next.js Frontend Dashboard on Port 3000...
cd "%~dp0frontend"
start "OceanEmbed Frontend (Port 3000)" cmd /c "npm run dev"

echo.
echo ==============================================================================
echo  OCEANEMBED IS LIVE AND RUNNING!
echo ==============================================================================
echo    Dashboard UI:   http://localhost:3000
echo    API Server:     http://localhost:8000
echo    API Docs:       http://localhost:8000/docs
echo ==============================================================================
echo.

