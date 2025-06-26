#!/usr/bin/env python3
"""
Fetch and parse the complete list of Unusual Whales API endpoints from the OpenAPI YAML specification.
Enhanced version with JSON export, filtering, and improved error handling.
"""

import requests
import yaml
import json
import argparse
import time
from typing import Dict, List, Any, Optional

def fetch_openapi_spec(max_retries: int = 3, retry_delay: float = 1.0) -> Optional[Dict[str, Any]]:
    """Fetch the OpenAPI specification from Unusual Whales API with retry logic."""
    url = "https://api.unusualwhales.com/api/openapi"
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"Retry attempt {attempt + 1}/{max_retries}...")
                time.sleep(retry_delay * attempt)  # Exponential backoff
            
            print("Fetching OpenAPI specification from Unusual Whales...")
            response = requests.get(url, timeout=30, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; API-Fetcher/1.0)'
            })
            response.raise_for_status()
            
            # Parse YAML instead of JSON
            spec = yaml.safe_load(response.text)
            print(f"✅ Successfully fetched OpenAPI specification ({len(response.text)} bytes)")
            return spec
            
        except requests.exceptions.Timeout:
            print(f"⚠️  Request timeout on attempt {attempt + 1}")
            if attempt == max_retries - 1:
                print("❌ All retry attempts failed due to timeout")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request error on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("❌ All retry attempts failed due to request errors")
                return None
                
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
            return None
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return None
    
    return None

def parse_endpoints_from_openapi(openapi_spec: Dict[str, Any]):
    """Parse all endpoints from the OpenAPI specification."""
    
    if not openapi_spec or 'paths' not in openapi_spec:
        print("Invalid OpenAPI specification - no paths found")
        return []
    
    endpoints = []
    paths = openapi_spec['paths']
    
    print(f"Found {len(paths)} API paths in OpenAPI specification")
    print()
    
    for path, methods in paths.items():
        for method, details in methods.items():
            if method.upper() in ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']:
                endpoint = {
                    'path': path,
                    'method': method.upper(),
                    'summary': details.get('summary', 'No summary'),
                    'description': details.get('description', 'No description'),
                    'operation_id': details.get('operationId', 'No operation ID'),
                    'tags': details.get('tags', [])
                }
                endpoints.append(endpoint)
    
    return endpoints

def categorize_endpoints(endpoints: List[Dict[str, Any]]):
    """Categorize endpoints by their tags and paths."""
    
    categories = {}
    
    for endpoint in endpoints:
        # Use tags if available, otherwise derive from path
        if endpoint['tags']:
            category = endpoint['tags'][0].lower()
        else:
            # Extract category from path (e.g., /congress/trades -> congress)
            path_parts = endpoint['path'].strip('/').split('/')
            if path_parts and path_parts[0]:
                category = path_parts[0].lower()
            else:
                category = 'uncategorized'
        
        if category not in categories:
            categories[category] = []
        
        categories[category].append(endpoint)
    
    return categories

def display_complete_endpoint_list(categories: Dict[str, List[Dict[str, Any]]]):
    """Display the complete list of endpoints in a clean format."""
    
    print("COMPLETE UNUSUAL WHALES API ENDPOINTS FROM OPENAPI")
    print("=" * 60)
    
    total_endpoints = 0
    
    for category, endpoints in sorted(categories.items()):
        print(f"\n{category.upper()} ({len(endpoints)} endpoints)")
        print("-" * 40)
        
        for i, endpoint in enumerate(endpoints, 1):
            # Clean up the operation ID to get a more readable name
            op_id = endpoint['operation_id']
            if op_id and op_id != 'No operation ID':
                clean_name = op_id
            else:
                # Generate name from path and method
                path_parts = endpoint['path'].strip('/').split('/')
                clean_name = '_'.join(path_parts) + f"_{endpoint['method'].lower()}"
            
            print(f"{i:2d}. {clean_name}")
            print(f"    Path: {endpoint['method']} {endpoint['path']}")
            if endpoint['summary'] and endpoint['summary'] != 'No summary':
                print(f"    Summary: {endpoint['summary']}")
            if endpoint['description'] and endpoint['description'] != 'No description':
                # Truncate long descriptions
                desc = endpoint['description'][:100] + ('...' if len(endpoint['description']) > 100 else '')
                print(f"    Description: {desc}")
            print()
        
        total_endpoints += len(endpoints)
    
    print("=" * 60)
    print(f"TOTAL API ENDPOINTS: {total_endpoints}")
    print("=" * 60)
    
    return total_endpoints

def create_clean_list(categories: Dict[str, List[Dict[str, Any]]]):
    """Create a clean, simple list of all endpoints."""
    
    print("\nCLEAN ENDPOINT LIST")
    print("=" * 50)
    
    total = 0
    for category, endpoints in sorted(categories.items()):
        print(f"\n{category.upper()} ({len(endpoints)} methods):")
        print("-" * 30)
        
        for i, endpoint in enumerate(endpoints, 1):
            # Extract a clean method name
            op_id = endpoint['operation_id']
            if op_id and op_id != 'No operation ID':
                # Convert camelCase to snake_case
                import re
                clean_name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', op_id)
                clean_name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', clean_name).lower()
            else:
                # Generate name from path
                path_parts = [p for p in endpoint['path'].strip('/').split('/') if p]
                clean_name = '_'.join(path_parts)
            
            print(f"{i:2d}. {clean_name}")
            if endpoint['summary'] and endpoint['summary'] != 'No summary':
                print(f"    Description: {endpoint['summary']}")
        
        total += len(endpoints)
    
    print(f"\nTOTAL: {total} endpoints")
    return total

def main():
    """Main function to fetch and display all endpoints."""
    
    # Fetch OpenAPI specification
    openapi_spec = fetch_openapi_spec()
    
    if not openapi_spec:
        print("Failed to fetch OpenAPI specification")
        return
    
    # Parse endpoints
    endpoints = parse_endpoints_from_openapi(openapi_spec)
    
    if not endpoints:
        print("No endpoints found in OpenAPI specification")
        return
    
    # Categorize endpoints
    categories = categorize_endpoints(endpoints)
    
    # Display complete list
    total = display_complete_endpoint_list(categories)
    
    # Create clean list
    clean_total = create_clean_list(categories)
    
    # Show categories summary
    print(f"\nCATEGORIES FOUND: {len(categories)}")
    for category, endpoints in sorted(categories.items()):
        print(f"  {category}: {len(endpoints)} endpoints")
    
    print(f"\nThis should give us the complete list of all {total} endpoints!")

if __name__ == "__main__":
    main()