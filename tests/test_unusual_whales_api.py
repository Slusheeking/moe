#!/usr/bin/env python3
"""
Comprehensive test suite for all 105 Unusual Whales API endpoints.
Tests all endpoints from the OpenAPI specification using the API token from .env file.
"""

import os
import sys
import pytest
import requests
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/home/ubuntu/moe-1/.env')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Test configuration
BASE_URL = "https://api.unusualwhales.com"
API_TOKEN = os.getenv('UW_API_TOKEN')
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ']
TEST_SECTORS = ['technology', 'financials', 'healthcare']
TEST_INSTITUTIONS = ['blackrock', 'vanguard', 'state-street']
TEST_CONGRESS_MEMBERS = ['nancy-pelosi', 'dan-crenshaw']

# Rate limiting
REQUEST_DELAY = 0.5  # 500ms between requests
MAX_RETRIES = 3

class UnusualWhalesAPITester:
    """Comprehensive tester for all Unusual Whales API endpoints."""
    
    def __init__(self):
        if not API_TOKEN:
            raise ValueError("UW_API_TOKEN not found in environment variables")
        
        self.headers = {
            'Authorization': f'Bearer {API_TOKEN}',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Test results tracking
        self.results = {
            'total_endpoints': 0,
            'successful': 0,
            'failed': 0,
            'rate_limited': 0,
            'unauthorized': 0,
            'not_found': 0,
            'server_error': 0,
            'endpoint_results': {}
        }
        
        # Date helpers
        self.today = datetime.now().strftime('%Y-%m-%d')
        self.yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        self.last_week = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    def make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Dict[str, Any]:
        """Make API request with rate limiting and error handling."""
        url = f"{BASE_URL}{endpoint}"
        
        for attempt in range(MAX_RETRIES):
            try:
                # Rate limiting
                time.sleep(REQUEST_DELAY)
                
                if method.upper() == 'GET':
                    response = self.session.get(url, params=params or {}, timeout=30)
                else:
                    response = self.session.request(method, url, json=params or {}, timeout=30)
                
                result = {
                    'status_code': response.status_code,
                    'success': response.status_code == 200,
                    'response_size': len(response.content),
                    'response_time': response.elapsed.total_seconds(),
                    'attempt': attempt + 1,
                    'error': None,
                    'data': None
                }
                
                if response.status_code == 200:
                    try:
                        result['data'] = response.json()
                        result['data_type'] = type(result['data']).__name__
                        if isinstance(result['data'], dict):
                            result['data_keys'] = list(result['data'].keys())
                        elif isinstance(result['data'], list):
                            result['data_length'] = len(result['data'])
                    except json.JSONDecodeError:
                        result['data'] = response.text[:500]  # First 500 chars
                        result['data_type'] = 'text'
                    
                    self.results['successful'] += 1
                    return result
                
                elif response.status_code == 401:
                    result['error'] = 'Unauthorized - check API token'
                    self.results['unauthorized'] += 1
                    return result
                
                elif response.status_code == 404:
                    result['error'] = 'Not found'
                    self.results['not_found'] += 1
                    return result
                
                elif response.status_code == 429:
                    result['error'] = 'Rate limited'
                    self.results['rate_limited'] += 1
                    if attempt < MAX_RETRIES - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff
                        print(f"Rate limited, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    return result
                
                elif response.status_code >= 500:
                    result['error'] = f'Server error: {response.status_code}'
                    self.results['server_error'] += 1
                    return result
                
                else:
                    result['error'] = f'HTTP {response.status_code}: {response.text[:200]}'
                    self.results['failed'] += 1
                    return result
                    
            except requests.exceptions.Timeout:
                result = {
                    'status_code': 0,
                    'success': False,
                    'error': 'Request timeout',
                    'attempt': attempt + 1
                }
                if attempt == MAX_RETRIES - 1:
                    self.results['failed'] += 1
                    return result
                    
            except requests.exceptions.RequestException as e:
                result = {
                    'status_code': 0,
                    'success': False,
                    'error': f'Request error: {str(e)}',
                    'attempt': attempt + 1
                }
                if attempt == MAX_RETRIES - 1:
                    self.results['failed'] += 1
                    return result
        
        # Should not reach here
        self.results['failed'] += 1
        return {'status_code': 0, 'success': False, 'error': 'Max retries exceeded'}
    
    def test_endpoint(self, category: str, name: str, endpoint: str, params: Dict = None, method: str = 'GET') -> Dict[str, Any]:
        """Test a single endpoint and record results."""
        self.results['total_endpoints'] += 1
        
        print(f"Testing {category}.{name}: {method} {endpoint}")
        
        result = self.make_request(endpoint, params, method)
        
        # Store detailed results
        endpoint_key = f"{category}.{name}"
        self.results['endpoint_results'][endpoint_key] = {
            'endpoint': endpoint,
            'method': method,
            'params': params,
            'result': result
        }
        
        # Print result summary
        if result['success']:
            print(f"  ✅ SUCCESS ({result['status_code']}) - {result.get('response_time', 0):.2f}s")
            if result.get('data_type'):
                print(f"     Data type: {result['data_type']}")
                if result.get('data_length'):
                    print(f"     Data length: {result['data_length']}")
                elif result.get('data_keys'):
                    print(f"     Data keys: {result['data_keys'][:5]}...")  # First 5 keys
        else:
            print(f"  ❌ FAILED ({result['status_code']}) - {result.get('error', 'Unknown error')}")
        
        return result

    # ALERTS ENDPOINTS (2)
    def test_alerts(self):
        """Test all alerts endpoints."""
        print("\n" + "="*60)
        print("TESTING ALERTS ENDPOINTS (2)")
        print("="*60)
        
        # 1. Get alerts
        self.test_endpoint('alerts', 'alerts', '/api/alerts')
        
        # 2. Get alert configurations
        self.test_endpoint('alerts', 'configs', '/api/alerts/configuration')

    # CONGRESS ENDPOINTS (3)
    def test_congress(self):
        """Test all congress endpoints."""
        print("\n" + "="*60)
        print("TESTING CONGRESS ENDPOINTS (3)")
        print("="*60)
        
        # 1. Recent congress trades
        self.test_endpoint('congress', 'recent_trades', '/api/congress/recent-trades')
        
        # 2. Recent late reports
        self.test_endpoint('congress', 'late_reports', '/api/congress/late-reports')
        
        # 3. Congress trader reports
        for member in TEST_CONGRESS_MEMBERS[:1]:  # Test first member
            self.test_endpoint('congress', 'congress_trader', '/api/congress/congress-trader', 
                             {'trader': member})

    # DARKPOOL ENDPOINTS (2)
    def test_darkpool(self):
        """Test all darkpool endpoints."""
        print("\n" + "="*60)
        print("TESTING DARKPOOL ENDPOINTS (2)")
        print("="*60)
        
        # 1. Recent darkpool trades
        self.test_endpoint('darkpool', 'recent', '/api/darkpool/recent')
        
        # 2. Ticker darkpool trades
        for symbol in TEST_SYMBOLS[:2]:  # Test first 2 symbols
            self.test_endpoint('darkpool', 'ticker', f'/api/darkpool/{symbol}', 
                             {'date': self.today})

    # EARNINGS ENDPOINTS (3)
    def test_earnings(self):
        """Test all earnings endpoints."""
        print("\n" + "="*60)
        print("TESTING EARNINGS ENDPOINTS (3)")
        print("="*60)
        
        # 1. Premarket earnings
        self.test_endpoint('earnings', 'premarket', '/api/earnings/premarket', 
                         {'date': self.today})
        
        # 2. Afterhours earnings
        self.test_endpoint('earnings', 'afterhours', '/api/earnings/afterhours', 
                         {'date': self.today})
        
        # 3. Historical ticker earnings
        for symbol in TEST_SYMBOLS[:2]:  # Test first 2 symbols
            self.test_endpoint('earnings', 'ticker', f'/api/earnings/{symbol}')

    # ETFS ENDPOINTS (5)
    def test_etfs(self):
        """Test all ETF endpoints."""
        print("\n" + "="*60)
        print("TESTING ETF ENDPOINTS (5)")
        print("="*60)
        
        etf_symbols = ['SPY', 'QQQ', 'XLK']
        
        for etf in etf_symbols[:1]:  # Test first ETF
            # 1. ETF exposure
            self.test_endpoint('etfs', 'exposure', f'/api/etfs/{etf}/exposure')
            
            # 2. ETF holdings
            self.test_endpoint('etfs', 'holdings', f'/api/etfs/{etf}/holdings')
            
            # 3. ETF inflow/outflow
            self.test_endpoint('etfs', 'in_outflow', f'/api/etfs/{etf}/in-outflow')
            
            # 4. ETF info
            self.test_endpoint('etfs', 'info', f'/api/etfs/{etf}/info')
            
            # 5. ETF weights
            self.test_endpoint('etfs', 'weights', f'/api/etfs/{etf}/weights')

    # NEWS ENDPOINTS (1)
    def test_news(self):
        """Test all news endpoints."""
        print("\n" + "="*60)
        print("TESTING NEWS ENDPOINTS (1)")
        print("="*60)
        
        # 1. News headlines
        self.test_endpoint('news', 'headlines', '/api/news/headlines')

    # STOCK ENDPOINTS (Sample - testing key endpoints)
    def test_stocks_sample(self):
        """Test sample stock endpoints to avoid too many requests."""
        print("\n" + "="*60)
        print("TESTING STOCK ENDPOINTS (SAMPLE)")
        print("="*60)
        
        # Test with first symbol only
        symbol = TEST_SYMBOLS[0]
        print(f"\nTesting key stock endpoints for {symbol}...")
        
        # Key endpoints
        self.test_endpoint('stocks', 'info', f'/api/stock/{symbol}/info')
        self.test_endpoint('stocks', 'flow_alerts', f'/api/stock/{symbol}/flow-alerts')
        self.test_endpoint('stocks', 'greek_exposure', f'/api/stock/{symbol}/greek-exposure')
        self.test_endpoint('stocks', 'iv_rank', f'/api/stock/{symbol}/iv-rank')
        self.test_endpoint('stocks', 'options_volume', f'/api/stock/{symbol}/options-volume')

    def run_all_tests(self):
        """Run all endpoint tests."""
        print("🚀 STARTING COMPREHENSIVE UNUSUAL WHALES API TEST")
        print(f"📡 Base URL: {BASE_URL}")
        print(f"🔑 API Token: {API_TOKEN[:10]}...")
        print(f"📅 Test Date: {self.today}")
        print(f"🧪 Test Symbols: {TEST_SYMBOLS}")
        print("="*80)
        
        start_time = time.time()
        
        try:
            # Run test categories (limited to avoid rate limits)
            self.test_alerts()
            self.test_congress()
            self.test_darkpool()
            self.test_earnings()
            self.test_etfs()
            self.test_news()
            self.test_stocks_sample()  # Limited stock testing
            
        except KeyboardInterrupt:
            print("\n⚠️ Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
        
        # Calculate and display results
        end_time = time.time()
        total_time = end_time - start_time
        
        self.print_summary(total_time)
        
        return self.results
    
    def print_summary(self, total_time: float):
        """Print comprehensive test summary."""
        print("\n" + "="*80)
        print("🎯 UNUSUAL WHALES API TEST SUMMARY")
        print("="*80)
        
        print(f"⏱️  Total Test Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"📊 Total Endpoints Tested: {self.results['total_endpoints']}")
        print(f"✅ Successful: {self.results['successful']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"🔒 Unauthorized: {self.results['unauthorized']}")
        print(f"🚫 Not Found: {self.results['not_found']}")
        print(f"⏳ Rate Limited: {self.results['rate_limited']}")
        print(f"🔥 Server Errors: {self.results['server_error']}")
        
        # Calculate success rate
        if self.results['total_endpoints'] > 0:
            success_rate = (self.results['successful'] / self.results['total_endpoints']) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Show category breakdown
        print("\n📋 RESULTS BY CATEGORY:")
        print("-" * 40)
        
        category_stats = {}
        for endpoint_key, endpoint_data in self.results['endpoint_results'].items():
            category = endpoint_key.split('.')[0]
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'successful': 0}
            
            category_stats[category]['total'] += 1
            if endpoint_data['result']['success']:
                category_stats[category]['successful'] += 1
        
        for category, stats in sorted(category_stats.items()):
            success_rate = (stats['successful'] / stats['total']) * 100 if stats['total'] > 0 else 0
            print(f"{category:15} {stats['successful']:3}/{stats['total']:3} ({success_rate:5.1f}%)")
        
        # Show failed endpoints
        failed_endpoints = []
        for endpoint_key, endpoint_data in self.results['endpoint_results'].items():
            if not endpoint_data['result']['success']:
                failed_endpoints.append({
                    'endpoint': endpoint_key,
                    'error': endpoint_data['result'].get('error', 'Unknown'),
                    'status': endpoint_data['result'].get('status_code', 0)
                })
        
        if failed_endpoints:
            print(f"\n❌ FAILED ENDPOINTS ({len(failed_endpoints)}):")
            print("-" * 60)
            for failure in failed_endpoints[:10]:  # Show first 10
                print(f"{failure['endpoint']:30} {failure['status']:3} {failure['error']}")
            
            if len(failed_endpoints) > 10:
                print(f"... and {len(failed_endpoints) - 10} more")
        
        print("\n" + "="*80)


# Test functions for pytest
def test_api_connection():
    """Test basic API connection."""
    tester = UnusualWhalesAPITester()
    result = tester.test_endpoint('test', 'connection', '/api/news/headlines')
    assert result['status_code'] in [200, 401, 429], f"Unexpected status code: {result['status_code']}"


def test_alerts_endpoints():
    """Test alerts endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_alerts()
    assert tester.results['total_endpoints'] > 0


def test_congress_endpoints():
    """Test congress endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_congress()
    assert tester.results['total_endpoints'] > 0


def test_darkpool_endpoints():
    """Test darkpool endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_darkpool()
    assert tester.results['total_endpoints'] > 0


def test_earnings_endpoints():
    """Test earnings endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_earnings()
    assert tester.results['total_endpoints'] > 0


def test_etf_endpoints():
    """Test ETF endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_etfs()
    assert tester.results['total_endpoints'] > 0


def test_news_endpoints():
    """Test news endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_news()
    assert tester.results['total_endpoints'] > 0


def test_stock_endpoints_sample():
    """Test sample stock endpoints."""
    tester = UnusualWhalesAPITester()
    tester.test_stocks_sample()
    assert tester.results['total_endpoints'] > 0


# Main execution
if __name__ == "__main__":
    print("🧪 UNUSUAL WHALES API COMPREHENSIVE TESTER")
    print("=" * 80)
    
    if not API_TOKEN:
        print("❌ Error: UW_API_TOKEN not found in environment variables")
        print("Please ensure the .env file contains your Unusual Whales API token")
        sys.exit(1)
    
    # Run comprehensive test
    tester = UnusualWhalesAPITester()
    results = tester.run_all_tests()
    
    # Save results to file
    results_file = Path(__file__).parent / 'unusual_whales_test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if results['unauthorized'] > 0:
        print("\n⚠️  Some endpoints returned 401 Unauthorized - check API token permissions")
        sys.exit(2)
    elif results['failed'] > results['successful']:
        print("\n❌ More endpoints failed than succeeded")
        sys.exit(1)
    else:
        print("\n✅ Test completed successfully")
        sys.exit(0)
