#!/usr/bin/env bash

# ==============================================================================
# OceanEmbed - Complete Full-Stack Launch Script (run.sh)
# ==============================================================================
# Starts:
#   1. FastAPI Python AI Inference Server (Port 8000)
#   2. Next.js High-Tech Operational Frontend (Port 3000)
# ==============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=============================================================================="
echo "🌊 LAUNCHING OCEANEMBED AI INTELLIGENCE PLATFORM (TRI-BREED v4.2)"
echo "=============================================================================="

# Trap SIGINT/SIGTERM to kill background children on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down OceanEmbed Full-Stack services..."
    if [ -n "$API_PID" ]; then
        kill "$API_PID" 2>/dev/null || true
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi
    echo "👋 All services stopped cleanly. Goodbye!"
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# 1. Start Python FastAPI Inference Engine
echo "🧠 [1/2] Starting Python PyTorch FastAPI Inference Server on Port 8000..."
./.venv/bin/python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --log-level info &
API_PID=$!

# Wait briefly for FastAPI to bind
sleep 2

# 2. Start Next.js Frontend
echo "🌐 [2/2] Starting Next.js Dashboard on Port 3000..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================================================================="
echo "🎉 OCEANEMBED IS LIVE AND READY!"
echo "=============================================================================="
echo "   🖥️  Dashboard UI:   http://localhost:3000"
echo "   ⚡ API Server:     http://localhost:8000"
echo "   📖 API Docs:       http://localhost:8000/docs"
echo "=============================================================================="
echo "Press Ctrl+C at any time to shut down all services."
echo ""

# Wait on background processes
wait "$FRONTEND_PID" "$API_PID"
