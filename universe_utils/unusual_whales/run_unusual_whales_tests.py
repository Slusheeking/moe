#!/usr/bin/env python3
"""
Simple test runner for Unusual Whales API tests.
Run this script to test all 99 REST API endpoints.
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='Run Unusual Whales API tests')
    parser.add_argument('--mode', choices=['all', 'critical', 'pytest'], default='all',
                       help='Test mode: all (comprehensive), critical (key endpoints only), or pytest (run with pytest)')
    parser.add_argument('--save-report', action='store_true', default=True,
                       help='Save detailed JSON report (default: True)')
    parser.add_argument('--ticker', default='AAPL',
                       help='Test ticker symbol (default: AAPL)')
    
    args = parser.parse_args()
    
    # Check for API token
    if not os.getenv('UW_API_TOKEN'):
        print("❌ Error: UW_API_TOKEN environment variable not found!")
        print("Please set your Unusual Whales API token:")
        print("export UW_API_TOKEN='your_token_here'")
        return 1
    
    print("🐋 Unusual Whales API Test Runner")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Test ticker: {args.ticker}")
    print(f"Save report: {args.save_report}")
    print("=" * 50)
    
    if args.mode == 'pytest':
        # Run with pytest
        import subprocess
        cmd = ['python', '-m', 'pytest', 'tests/test_unusual_whales_api.py', '-v']
        return subprocess.call(cmd)
    
    else:
        # Run directly
        try:
            from tests.test_unusual_whales_api import UnusualWhalesAPITester
            
            tester = UnusualWhalesAPITester()
            
            # Override test ticker if specified
            if args.ticker != 'AAPL':
                tester.test_ticker = args.ticker
            
            if args.mode == 'critical':
                # Test only critical endpoints
                print("🎯 Running critical endpoints test...")
                
                critical_endpoints = [
                    ("/api/news/headlines", {}),
                    (f"/api/stock/{tester.test_ticker}/info", {}),
                    (f"/api/stock/{tester.test_ticker}/flow-recent", {}),
                    ("/api/market/sector-etfs", {}),
                    (f"/api/darkpool/{tester.test_ticker}", {'date': tester.past_date}),
                ]
                
                results = {}
                for endpoint, params in critical_endpoints:
                    print(f"Testing {endpoint}...")
                    result = tester.make_request(endpoint, params=params)
                    results[endpoint] = result
                    status = "✅" if result['success'] else "❌"
                    print(f"  {status} {endpoint}: {result['status_code']}")
                
                # Summary
                successful = sum(1 for r in results.values() if r['success'])
                total = len(results)
                print(f"\n📊 Critical endpoints: {successful}/{total} successful")
                
                if successful == total:
                    print("🎉 All critical endpoints working!")
                    return 0
                else:
                    print("⚠️  Some critical endpoints failed")
                    return 1
            
            else:
                # Run comprehensive test
                print("🚀 Running comprehensive API test suite...")
                results = tester.run_all_tests()
                
                if args.save_report:
                    tester.save_detailed_report()
                
                # Return success/failure based on results
                successful_tests = sum(1 for result in results.values() if result['success'])
                total_tests = len(results)
                success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
                
                if success_rate >= 70:  # 70% success threshold
                    print("🎉 Test suite completed successfully!")
                    return 0
                else:
                    print("⚠️  Test suite completed with issues")
                    return 1
                    
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    exit(main())
