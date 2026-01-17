# 🔨 How to Compile Your EA for Strategy Testing

## 🎯 **The Problem**

Strategy Tester **ONLY shows compiled EAs** (`.ex5` files), not source code (`.mq5` files).

You have: `AdvanceEA.mq5` (source code) ✓
You need: `AdvanceEA.ex5` (compiled executable) ❌

---

## ✅ **Solution: Compile in MetaEditor**

### **Step-by-Step Instructions:**

#### **1. Open MetaEditor**
   - In MT5, press **F4**
   - Or click: **Tools → MetaQuotes Language Editor**

#### **2. Find Your EA**
   - In MetaEditor's **Navigator** panel (left side)
   - Expand: **Expert Advisors**
   - Look for: **AdvanceEA**
   
   **If you DON'T see it:**
   - Click: **File → Open**
   - Navigate to: **MQL5/Experts/**
   - Select: **AdvanceEA.mq5**
   - Click **Open**

#### **3. Compile the EA**
   - With `AdvanceEA.mq5` open in the editor
   - Press **F7** (or click the **Compile** button)
   - Watch the **Toolbox** panel at the bottom

#### **4. Check for Errors**
   In the **Errors** tab (bottom panel), you should see:
   ```
   0 error(s), 0 warning(s)
   ```
   
   **If you see errors:**
   - Read the error messages
   - They'll tell you what's wrong (usually line numbers)
   - Fix the errors and compile again (F7)

#### **5. Verify Compilation Success**
   Look for this message:
   ```
   AdvanceEA.ex5    X KB    Successfully compiled
   ```

#### **6. Check the File Was Created**
   - In MetaEditor: **File → Open Data Folder**
   - Navigate to: **MQL5/Experts/**
   - You should now see TWO files:
     - `AdvanceEA.mq5` (source code)
     - `AdvanceEA.ex5` (compiled - NEW!)

---

## 🎮 **Now Open Strategy Tester**

### **After successful compilation:**

1. **Go back to MT5** (not MetaEditor)

2. **Open Strategy Tester**
   - Press **Ctrl+R**
   - Or: **View → Strategy Tester**

3. **Select Your EA**
   - In the **Expert Advisor** dropdown
   - You should now see: **AdvanceEA**
   - Select it!

4. **Configure Test Settings**
   - **Symbol**: Choose your pair (e.g., EURUSD)
   - **Timeframe**: Choose period (e.g., H1)
   - **Date Range**: Set start and end dates
   - **Execution Mode**: Choose model (usually "Every tick" for accuracy)
   - **Deposit**: Set initial balance (e.g., 10000)

5. **Start the Test**
   - Click the **Start** button
   - Watch the test run!

---

## 🐛 **Troubleshooting**

### **EA Still Doesn't Appear in Strategy Tester?**

1. **Make sure compilation succeeded** (0 errors)
2. **Restart MT5 completely**
   ```bash
   pkill -f terminal64.exe
   ~/mt5.sh
   ```
3. **Check if .ex5 file exists**
   - In MetaEditor: File → Open Data Folder
   - Go to: MQL5/Experts/
   - Look for: AdvanceEA.ex5

4. **Refresh in MT5**
   - In Navigator: Right-click "Expert Advisors" → Refresh
   - In Strategy Tester: Reopen it (Ctrl+R again)

### **Common Compilation Errors**

**Error: "undeclared identifier"**
- Solution: Check variable names, make sure all functions are defined

**Error: "syntax error"**
- Solution: Check for missing semicolons, brackets, etc.

**Error: "function not defined"**
- Solution: Make sure all required functions exist

---

## 📊 **File Types Explained**

| Extension | Type | Visible In | Can Edit? | Can Run? |
|-----------|------|------------|-----------|----------|
| `.mq5` | Source code | MetaEditor | ✓ Yes | ❌ No |
| `.ex5` | Compiled executable | MT5 Navigator & Strategy Tester | ❌ No | ✓ Yes |

**Remember:** 
- Edit the `.mq5` file
- Compile to create `.ex5` file
- MT5 runs the `.ex5` file

---

## ⚡ **Quick Reference**

| Task | Keyboard Shortcut | Menu Path |
|------|-------------------|-----------|
| Open MetaEditor | F4 | Tools → MetaQuotes Language Editor |
| Compile | F7 | Compile button |
| Open Strategy Tester | Ctrl+R | View → Strategy Tester |
| Open Data Folder | Ctrl+Shift+D | File → Open Data Folder |

---

## 🎯 **Next Steps**

1. Open MetaEditor (F4)
2. Find AdvanceEA.mq5
3. Compile it (F7)
4. Check for 0 errors
5. Go to MT5 Strategy Tester (Ctrl+R)
6. Select AdvanceEA from dropdown
7. Start testing!

**Let me know if you get any compilation errors and I'll help fix them!** 🚀
