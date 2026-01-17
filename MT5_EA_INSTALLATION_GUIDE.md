# 🚀 MT5 Expert Advisor Installation Guide

## ❌ Problem: Can't See Your EA in MT5 Navigator

Your EA file is copied to multiple locations, but MT5 might be using a different folder.

---

## ✅ **SOLUTION: Use MT5's Built-in "Open Data Folder"**

This is the **GUARANTEED** way to find where MT5 actually stores files.

### **Step-by-Step Instructions:**

1. **Open MetaTrader 5**
   ```bash
   ~/mt5.sh
   ```

2. **Find the ACTUAL Data Folder**
   - In MT5 menu: **File → Open Data Folder**
   - OR press: **Ctrl+Shift+D**
   - A file explorer window will open showing your MT5 data folder

3. **Navigate to Experts Folder**
   - In the opened folder, go to: **MQL5 → Experts**

4. **Copy Your EA There**
   - Open a new terminal
   - Run this command (replace `/path/shown/in/explorer` with the actual path from step 2):
   ```bash
   cp ~/AdvanceEA.mq5 "/path/from/file/explorer/MQL5/Experts/"
   ```

5. **Refresh MT5 Navigator**
   - Back in MT5, open Navigator (Ctrl+N)
   - Right-click on **"Expert Advisors"**
   - Click **"Refresh"**

6. **Your EA should now appear!**

---

## 🔨 **Alternative: Compile in MetaEditor**

1. **Open MetaEditor** (Press F4 in MT5)

2. **Find Data Folder in MetaEditor**
   - In MetaEditor: **File → Open Data Folder**

3. **Navigate to Experts**
   - Go to: **MQL5 → Experts**

4. **Check if AdvanceEA.mq5 is there**
   - If YES: Open it and press F7 to compile
   - If NO: Copy it there using the terminal command above

5. **Compile**
   - Open **AdvanceEA.mq5** in MetaEditor
   - Press **F7** or click **Compile** button
   - Check for **0 errors** in the output

6. **Return to MT5**
   - Your EA should now appear in Navigator

---

## 📁 **Where We've Already Copied Your EA**

Your EA has been copied to these locations:
- ✓ `Program Files/MetaTrader 5/MQL5/Experts/`
- ✓ `AppData/Roaming/MetaQuotes/Terminal/9F3B8A2E4C6D1F8A7B5C3D2E1F9A8B7C/MQL5/Experts/`
- ✓ `AppData/Roaming/MetaQuotes/Terminal/*/MQL5/Experts/`

But **MT5 might be using a different folder!** That's why we need to use the "Open Data Folder" method.

---

## 🎯 **Quick Commands**

### Install EA to all locations:
```bash
./find_active_mt5_folder.sh
```

### Start MT5:
```bash
~/mt5.sh
```

### Copy EA manually (if you know the exact path):
```bash
cp ~/AdvanceEA.mq5 "/exact/path/from/open/data/folder/MQL5/Experts/"
```

---

## 🐛 **Still Not Working?**

1. **Check if Navigator is visible**
   - Press **Ctrl+N** to show/hide Navigator panel

2. **Check if you're looking in the right section**
   - In Navigator, expand: **Expert Advisors** (not Scripts or Indicators)

3. **Verify file permissions**
   ```bash
   ls -lh ~/AdvanceEA.mq5
   ```

4. **Check for compilation errors**
   - If the EA appears but won't run, it might have compilation errors
   - Open it in MetaEditor (F4) and compile (F7)
   - Check the "Errors" tab for any issues

5. **Restart MT5 completely**
   - Close all MT5 windows
   - Run: `pkill -f terminal64.exe`
   - Start fresh: `~/mt5.sh`

---

## 📞 **Next Steps**

After following the "Open Data Folder" method above, please tell me:
1. What path does "Open Data Folder" show?
2. Is AdvanceEA.mq5 in the MQL5/Experts/ folder there?
3. Can you see it in Navigator after refreshing?

This will help us identify the exact issue!
