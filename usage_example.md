# Cancel Orders and Positions Script Usage

## Quick Examples

### Cancel orders and close position for a specific symbol:
```bash
python trading_system/cancel_orders_positions.py --symbol QNTM
```

### Cancel ALL orders and close ALL positions:
```bash
python trading_system/cancel_orders_positions.py --symbol ALL
```

## What it does:

### For specific symbol (e.g., QNTM):
1. ✅ Cancels all open orders for QNTM
2. ✅ Closes the position by placing market sell order
3. ✅ Shows detailed summary

### For ALL:
1. ✅ Cancels ALL open orders 
2. ✅ Closes ALL positions with market sell orders
3. ✅ Requires "YES" confirmation for safety
4. ✅ Shows comprehensive summary

## Safety Features:
- ⚠️ Confirmation required for ALL operations
- 📊 Shows position details before closing
- 📋 Lists orders before canceling  
- ✅ Detailed success/failure reporting
- 🔒 Validates symbol format

## Example Output:
```
🔍 Checking orders for QNTM...
📋 Found 2 open orders for QNTM
  Cancelling: trailing_stop 366 QNTM (ID: 12345)
  ✅ Cancelled order 12345
🎯 QNTM Orders: 1 cancelled, 0 failed

🔍 Checking position for QNTM...
📊 Found position: 366 shares of QNTM @ $35.76 ($13,088.16)
  Placing market sell order for 366 shares...
  ✅ Market sell order placed for QNTM

==================================================
📋 FINAL SUMMARY FOR QNTM:
  Orders cancelled: 1
  Orders failed: 0
  Positions closed: 1
  Positions failed: 0
==================================================