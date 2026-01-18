#!/bin/bash
# Monitor and keep the feedback loop system running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/sam_feedback_loop.log"
PID_FILE="/tmp/sam_feedback_loop.pid"

echo "🔄 Feedback Loop Monitor Started"
echo "   Monitoring: sam_gemini_voice.py"
echo "   Log: $LOG_FILE"
echo "   Press Ctrl+C to stop"
echo ""

# Function to check if process is running
check_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to start the application
start_app() {
    echo "[$(date '+%H:%M:%S')] 🚀 Starting application..."
    cd "$SCRIPT_DIR"
    /Users/ajayraj/miniconda3/envs/JuneBrainEyeTracker/bin/python sam_gemini_voice.py > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
    if check_process; then
        echo "[$(date '+%H:%M:%S')] ✅ Application started (PID: $(cat $PID_FILE))"
    else
        echo "[$(date '+%H:%M:%S')] ❌ Failed to start application"
    fi
}

# Function to show feedback loop activity
show_feedback() {
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 Recent Feedback Loop Activity:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -20 "$LOG_FILE" | grep -E "Feedback Loop|Auto-corrected|Optimized|Accuracy" || echo "   (Waiting for feedback activity...)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
    fi
}

# Start the application
start_app

# Monitor loop
while true; do
    sleep 10
    
    if ! check_process; then
        echo "[$(date '+%H:%M:%S')] ⚠️  Process stopped, restarting..."
        start_app
    fi
    
    # Show feedback every 30 seconds
    if [ $(($(date +%S) % 30)) -eq 0 ]; then
        show_feedback
    fi
done

