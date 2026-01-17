#!/bin/bash
# Script to find the active MT5 terminal folder and install EA

echo "🔍 Finding active MT5 terminal folders..."
echo ""

# Check for recently modified files (last 24 hours)
echo "📊 Recently active terminal folders (modified in last 24 hours):"
find ~/.wine/drive_c -path "*/MetaQuotes/Terminal/*/MQL5" -type d -mtime -1 2>/dev/null | while read -r folder; do
    terminal_folder=$(dirname "$folder")
    echo "  📁 $terminal_folder"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Installing AdvanceEA.mq5 to ALL possible locations..."
echo ""

# Install to Program Files location
if [ -d ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/MQL5/Experts ]; then
    cp ~/AdvanceEA.mq5 ~/.wine/drive_c/Program\ Files/MetaTrader\ 5/MQL5/Experts/
    echo "✓ Copied to: Program Files/MetaTrader 5/MQL5/Experts/"
fi

# Install to all terminal folders
find ~/.wine/drive_c -path "*/Terminal/*/MQL5/Experts" -type d 2>/dev/null | while read -r folder; do
    cp ~/AdvanceEA.mq5 "$folder/"
    echo "✓ Copied to: $folder"
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Installation complete!"
echo ""
echo "🎯 IMPORTANT: Follow these steps IN MetaTrader 5:"
echo ""
echo "   METHOD 1 - Find your actual data folder:"
echo "   =========================================="
echo "   1. In MT5, click: File → Open Data Folder"
echo "   2. This opens the ACTUAL folder MT5 is using"
echo "   3. Navigate to: MQL5/Experts/"
echo "   4. Check if AdvanceEA.mq5 is there"
echo "   5. If NOT, copy it there manually"
echo ""
echo "   METHOD 2 - Compile in MetaEditor:"
echo "   ================================="
echo "   1. Press F4 to open MetaEditor"
echo "   2. Click: File → Open Data Folder (in MetaEditor too)"
echo "   3. Go to: MQL5/Experts/"
echo "   4. Copy AdvanceEA.mq5 there if missing"
echo "   5. In MetaEditor: File → Open → AdvanceEA.mq5"
echo "   6. Press F7 to compile"
echo ""
echo "   METHOD 3 - Check Navigator:"
echo "   ==========================="
echo "   1. In MT5 Navigator panel (Ctrl+N if hidden)"
echo "   2. Right-click 'Expert Advisors' → Refresh"
echo "   3. Look for AdvanceEA in the list"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
