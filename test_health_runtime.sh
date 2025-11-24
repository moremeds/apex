#!/bin/bash
# Test script to run the application and capture health component logs

echo "Starting application with debug logging..."
echo "Press Ctrl+C after a few seconds to stop"
echo "==============================================="

# Run for 5 seconds then kill
timeout 5s python main.py --env dev --no-dashboard 2>&1 | grep -E "(health|Health|component|Component)" | head -30

echo ""
echo "==============================================="
echo "Test complete. Check logs above for health component registration."
