#!/bin/bash
# Helper script to install AdvanceEA.mq5 to all MT5 Experts folders

EA_FILE=~/AdvanceEA.mq5

if [ ! -f "$EA_FILE" ]; then
    echo "❌ Error: AdvanceEA.mq5 not found in home directory!"
    exit 1
fi

echo "📂 Searching for MT5 Experts folders..."
echo ""

# Find all MT5 Experts folders
EXPERTS_FOLDERS=$(find ~/.wine/drive_c -path "*/MetaQuotes/Terminal/*/MQL5/Experts" -type d 2>/dev/null)

if [ -z "$EXPERTS_FOLDERS" ]; then
    echo "❌ No MT5 Experts folders found!"
    echo ""
    echo "Please make sure MT5 is installed via Wine."
    exit 1
fi

COUNT=0
echo "$EXPERTS_FOLDERS" | while read -r folder; do
    if [ -n "$folder" ]; then
        COUNT=$((COUNT + 1))
        echo "📁 Found: $folder"
        cp "$EA_FILE" "$folder/"
        if [ $? -eq 0 ]; then
            echo "   ✓ Copied AdvanceEA.mq5"
        else
            echo "   ❌ Failed to copy"
        fi
        echo ""
    fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation complete!"
echo ""
echo "📋 Next steps in MetaTrader 5:"
echo "   1. In Navigator panel, right-click on 'Expert Advisors'"
echo "   2. Select 'Refresh'"
echo "   3. Or press F4 to open MetaEditor"
echo "   4. Navigate to MQL5/Experts/AdvanceEA.mq5"
echo "   5. Press F7 to compile"
echo ""
echo "💡 Tip: If you still don't see it, restart MT5 completely"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
