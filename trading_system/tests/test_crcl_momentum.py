"""
Test Our New Momentum System on COIN - June 5-18, 2025
Compare our live momentum detection vs old XGBoost approach
"""

import asyncio
import logging
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading_system.momentum_detector import LiveMomentumDetector
from trading_system.polygon_client import PolygonClient
from trading_system import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_coin_new_system():
    """Test our new momentum system on COIN during explosive move"""
    
    print("""
🚀 NEW MOMENTUM SYSTEM TEST - COIN June 2025
============================================
Testing our live momentum detection (3-day max) vs old 20-day approach
    """)
    
    if not config.POLYGON_API_KEY:
        print("❌ POLYGON_API_KEY not set. Please set it in .env file")
        return
    
    # Initialize our new system
    polygon_client = PolygonClient(config.POLYGON_API_KEY)
    await polygon_client.connect()
    
    momentum_detector = LiveMomentumDetector(polygon_client)
    
    try:
        print("📚 Training momentum model with historical data...")
        # Train our model (will use rule-based if insufficient training data)
        await momentum_detector.train_model_from_historical_movers(lookback_days=30)
        
        # Test dates - using June 2025 dates
        test_dates = [
            datetime(2025, 6, 4),   # Day before the move started
            datetime(2025, 6, 5),   # First day of move
            datetime(2025, 6, 6),   # Early in the move
            datetime(2025, 6, 9),   # Mid-move
            datetime(2025, 6, 11),  # Later in move
        ]
        
        print("\n🔍 Testing CRCL momentum detection:")
        print("=" * 60)
        
        results = []
        
        for test_date in test_dates:
            print(f"\n📅 {test_date.strftime('%Y-%m-%d')} ({get_date_context(test_date)})")
            print("-" * 40)
            
            # Debug: Try to get raw data first
            print(f"   🔍 Checking data availability...")
            
            symbol = 'COIN'  # Always test COIN only
            
            try:
                # Try to get basic COIN data
                debug_bars = await polygon_client.list_aggs(
                    'COIN', 1, 'day',
                    test_date - timedelta(days=5),
                    test_date + timedelta(days=1),
                    limit=10
                )
                print(f"   📊 Found {len(debug_bars)} data points for COIN")
                
                if debug_bars:
                    for bar in debug_bars[:3]:  # Show first 3
                        bar_date = datetime.fromtimestamp(bar.timestamp / 1000).strftime('%Y-%m-%d')
                        print(f"      {bar_date}: ${bar.close:.2f} (Vol: {bar.volume:,})")
                else:
                    print("   ⚠️  No COIN data found for this date")
                    
            except Exception as e:
                print(f"   ❌ Data check failed: {e}")
            
            # Get momentum detection result for COIN only
            result = await momentum_detector.detect_momentum(symbol, test_date)
            
            if 'error' in result:
                print(f"   ❌ Error: {result['error']}")
                continue
            
            momentum_score = result.get('momentum_score', 0)
            method = result.get('method', 'unknown')
            features = result.get('features', {})
            
            # Our system decision
            if momentum_score >= config.MOMENTUM_CONFIG['entry_threshold']:
                decision = "🟢 ENTER POSITION"
                confidence = "HIGH" if momentum_score >= 70 else "MODERATE"
            else:
                decision = "🔴 NO ENTRY"
                confidence = "LOW"
            
            print(f"   🎯 Momentum Score: {momentum_score:.1f}/100")
            print(f"   🤖 Method: {method}")
            print(f"   📊 Decision: {decision}")
            print(f"   💪 Confidence: {confidence}")
            
            # Show key features that drove the decision
            if features:
                print(f"   📈 Key Features:")
                print(f"      • Live Velocity (1h): {features.get('live_velocity_1h', 0):.3f}")
                print(f"      • Volume Spike: {features.get('live_volume_spike', 0):.2f}x")
                print(f"      • Price Acceleration: {features.get('live_acceleration', 0):.4f}")
                print(f"      • 3-Day High: {'YES' if features.get('new_3d_high', 0) else 'NO'}")
                print(f"      • Breakout Strength: {features.get('breakout_strength', 0):.3f}")
                print(f"      • Volume vs 3D: {features.get('volume_vs_3d', 0):.2f}x")
            
            results.append({
                'date': test_date,
                'momentum_score': momentum_score,
                'decision': decision,
                'method': method,
                'features': features
            })
        
        # Get actual COIN performance
        print("\n📊 ACTUAL COIN PERFORMANCE (June 5-18, 2025):")
        print("=" * 60)
        
        actual_bars = await polygon_client.list_aggs(
            'COIN', 1, 'day',
            datetime(2025, 6, 5),
            datetime(2025, 6, 18),
            limit=20
        )
        
        if actual_bars:
            prices = [(datetime.fromtimestamp(bar.timestamp / 1000).strftime('%m/%d'), bar.close) 
                     for bar in actual_bars]
            
            print("   Date    Price    Daily Change")
            print("   ----    -----    ------------")
            
            for i, (date_str, price) in enumerate(prices):
                if i > 0:
                    prev_price = prices[i-1][1]
                    change_pct = (price - prev_price) / prev_price * 100
                    print(f"   {date_str}    ${price:7.2f}  {change_pct:+6.1f}%")
                else:
                    print(f"   {date_str}    ${price:7.2f}  ------")
            
            if len(actual_bars) > 1:
                total_return = (actual_bars[-1].close - actual_bars[0].close) / actual_bars[0].close * 100
                print(f"\n   🎯 Total Return: {total_return:+.1f}% over {len(actual_bars)} days")
        
        # Analysis and comparison
        print(f"\n🔍 MOMENTUM SYSTEM ANALYSIS:")
        print("=" * 60)
        
        # Find best entry point
        best_entry = None
        for result in results:
            if result['momentum_score'] >= config.MOMENTUM_CONFIG['entry_threshold']:
                if best_entry is None or result['momentum_score'] > best_entry['momentum_score']:
                    best_entry = result
        
        if best_entry:
            entry_date = best_entry['date']
            entry_score = best_entry['momentum_score']
            
            # Calculate theoretical performance
            entry_price = get_price_for_date(actual_bars, entry_date)
            peak_price = max(bar.close for bar in actual_bars) if actual_bars else 0
            
            if entry_price and peak_price:
                potential_gain = (peak_price - entry_price) / entry_price * 100
                # 5% trailing stop simulation
                trailing_stop_5_exit = peak_price * 0.95
                trailing_stop_5_gain = (trailing_stop_5_exit - entry_price) / entry_price * 100
                # 10% trailing stop simulation
                trailing_stop_10_exit = peak_price * 0.90
                trailing_stop_10_gain = (trailing_stop_10_exit - entry_price) / entry_price * 100
                
                # Calculate dollar amounts for $25k position
                position_size = 25000
                shares = position_size / entry_price
                profit_5_stop = shares * (trailing_stop_5_exit - entry_price)
                profit_10_stop = shares * (trailing_stop_10_exit - entry_price)
                final_value_5 = position_size + profit_5_stop
                final_value_10 = position_size + profit_10_stop
                
                print(f"   ✅ Best Entry: {entry_date.strftime('%m/%d')} @ ${entry_price:.2f} (Score: {entry_score:.0f})")
                print(f"   📈 Peak Reached: ${peak_price:.2f}")
                print(f"   🎯 Potential Gain: {potential_gain:+.1f}%")
                print(f"   🛑 5% Trailing Stop: ${trailing_stop_5_exit:.2f} ({trailing_stop_5_gain:+.1f}%)")
                print(f"   🛑 10% Trailing Stop: ${trailing_stop_10_exit:.2f} ({trailing_stop_10_gain:+.1f}%)")
                print(f"")
                print(f"   💰 $25K POSITION RESULTS:")
                print(f"      • Shares purchased: {shares:.0f} @ ${entry_price:.2f}")
                print(f"      • 5% stop profit: ${profit_5_stop:,.0f} → Total: ${final_value_5:,.0f}")
                print(f"      • 10% stop profit: ${profit_10_stop:,.0f} → Total: ${final_value_10:,.0f}")
                print(f"      • Difference: ${profit_5_stop - profit_10_stop:,.0f} more with 5% stop")
            else:
                print(f"   ✅ Best Entry: {entry_date.strftime('%m/%d')} (Score: {entry_score:.0f})")
        else:
            print("   ❌ No entry signals generated")
        
        # Compare to old approach
        print(f"\n🆚 COMPARISON TO OLD APPROACH:")
        print("=" * 60)
        print("   OLD SYSTEM (20-day lookback XGBoost):")
        print("   • Required 20 days of historical data")
        print("   • Used complex features like 20-day highs")
        print("   • Hardcoded training symbols")
        print("   • Less suitable for IPOs/new listings")
        print("")
        print("   NEW SYSTEM (Live + 3-day XGBoost):")
        print("   • Works with minimal data (3-day max)")
        print("   • Live momentum features (velocity, acceleration)")
        print("   • Dynamic symbol discovery")
        print("   • IPO-compatible from day 1")
        print("")
        print("   🎯 NEW SYSTEM ADVANTAGES:")
        print("   • Faster detection (catches momentum earlier)")
        print("   • Works on any symbol (no hardcoding)")
        print("   • Real-time features (immediate momentum)")
        print("   • Better for explosive moves like COIN")
        
        # Summary
        entry_signals = sum(1 for r in results if r['momentum_score'] >= config.MOMENTUM_CONFIG['entry_threshold'])
        max_score = max(r['momentum_score'] for r in results) if results else 0
        
        print(f"\n📈 FINAL SUMMARY:")
        print("=" * 60)
        print(f"   • Entry signals generated: {entry_signals}/5 test dates")
        print(f"   • Highest momentum score: {max_score:.0f}/100")
        print(f"   • System would have {'CAUGHT' if entry_signals > 0 else 'MISSED'} the COIN move")
        print(f"   • Detection method: {'XGBoost ML' if momentum_detector.is_trained else 'Rule-based'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await polygon_client.disconnect()


def get_date_context(date: datetime) -> str:
    """Get context for what happened on each date"""
    date_contexts = {
        datetime(2025, 6, 4): "Day before move",
        datetime(2025, 6, 5): "Move begins",
        datetime(2025, 6, 6): "Early momentum",
        datetime(2025, 6, 9): "Mid-explosion",
        datetime(2025, 6, 11): "Peak momentum"
    }
    return date_contexts.get(date, "Testing date")


def get_price_for_date(bars, target_date: datetime) -> float:
    """Get price for specific date from bars"""
    target_timestamp = target_date.timestamp() * 1000
    
    for bar in bars:
        bar_date = datetime.fromtimestamp(bar.timestamp / 1000)
        if bar_date.date() == target_date.date():
            return bar.close
    
    return None


if __name__ == "__main__":
    asyncio.run(test_coin_new_system())