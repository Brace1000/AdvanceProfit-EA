# ⚡ Quick Start: Strategy Testing in 5 Steps

## 🎯 Why Can't I See My EA in Strategy Tester?

**Answer:** You have the source code (`.mq5`) but need the compiled version (`.ex5`)

---

## 📋 **5 Simple Steps:**

### **STEP 1: Open MetaEditor**
```
In MT5 → Press F4
```

### **STEP 2: Open Your EA**
```
In MetaEditor Navigator (left panel):
→ Expand "Expert Advisors"
→ Double-click "AdvanceEA"

(If not visible: File → Open → MQL5/Experts/AdvanceEA.mq5)
```

### **STEP 3: Compile**
```
Press F7 (or click Compile button)
```

### **STEP 4: Check Result**
```
Bottom panel should show:
✓ 0 error(s), 0 warning(s)
✓ AdvanceEA.ex5 successfully compiled
```

### **STEP 5: Open Strategy Tester**
```
Back in MT5 → Press Ctrl+R
→ Select "AdvanceEA" from dropdown
→ Configure settings
→ Click START
```

---

## 🔍 **Visual Checklist**

Before Strategy Testing:
- [ ] AdvanceEA.mq5 exists in Experts folder ✓ (You have this)
- [ ] AdvanceEA.ex5 created after compilation ❌ (You need this)
- [ ] 0 compilation errors ❓ (Check this)
- [ ] EA appears in Strategy Tester dropdown ❓ (Will appear after compilation)

---

## 🆘 **If You Get Compilation Errors:**

1. **Read the error message** - it tells you the line number
2. **Common fixes:**
   - Missing semicolon `;`
   - Mismatched brackets `{ }`
   - Undefined variables
3. **Copy the error message** and ask for help

---

## 🎮 **Strategy Tester Settings (Recommended):**

| Setting | Recommended Value |
|---------|------------------|
| Symbol | EURUSD |
| Timeframe | H1 (1 Hour) |
| Dates | Last 1-3 months |
| Execution | Every tick based on real ticks |
| Deposit | 10000 |
| Leverage | 1:100 or 1:500 |

---

## ✅ **Success Looks Like:**

```
MetaEditor Compile Output:
├── 0 error(s), 0 warning(s)
└── AdvanceEA.ex5    42 KB    Successfully compiled

MT5 Strategy Tester:
├── Expert Advisor: [AdvanceEA]  ← You should see this!
├── Symbol: EURUSD
└── [START] button ready to click
```

---

**That's it! Press F4 → F7 → Ctrl+R and you're testing!** 🚀
