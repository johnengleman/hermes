#!/bin/bash

echo "Checking optimizer status..."
echo ""

# Check if process is still running
if ps aux | grep -v grep | grep "optimize_hermes.py" > /dev/null; then
    echo "✅ Optimizer is STILL RUNNING"
    ps aux | grep -v grep | grep "optimize_hermes.py" | awk '{print "  CPU: "$3"% | Memory: "$4"% | Runtime: "$10}'
    echo ""
    echo "Output files not ready yet. Check back in a few minutes."
else
    echo "✅ Optimizer COMPLETED!"
    echo ""

    # Show the results
    if [ -f "hermes_quick_BTC.csv" ]; then
        echo "=== BTC Results ==="
        head -30 hermes_quick_BTC.csv
    else
        echo "⚠️  No output file found yet"
    fi
fi
