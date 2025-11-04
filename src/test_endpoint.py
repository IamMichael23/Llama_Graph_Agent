"""
Endpoint Diagnostic Test Script

This script tests the custom API endpoint to diagnose timeout issues.
It will:
1. Test network connectivity to api.agicto.cn
2. Test LLM endpoint with 'gpt-5-nano' model
3. Test embedding endpoint with 'text-embedding-3-small'
4. Measure response times for each
5. Identify specific timeout issues
"""

import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

API_KEY = os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("OPENAI_API_BASE")
EMBEDDING_KEY = os.getenv("EMBEDDING_KEY")

print("="*70)
print("🔍 ENDPOINT DIAGNOSTIC TEST")
print("="*70)
print(f"⏱️  Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🌐 API Base URL: {API_BASE}")
print(f"🔑 API Key loaded: {'Yes' if API_KEY else 'No'}")
print(f"🔑 Embedding Key loaded: {'Yes' if EMBEDDING_KEY else 'No'}")
print("="*70)

# Test Results Storage
results = {
    "network_test": None,
    "llm_test": None,
    "embedding_test": None
}


def test_network_connectivity():
    """Test basic network connectivity to the endpoint"""
    print("\n" + "="*70)
    print("TEST 1: Network Connectivity")
    print("="*70)

    try:
        print(f"🌐 Testing connection to {API_BASE}...")
        start_time = time.time()

        # Simple GET request to check if endpoint is reachable
        response = requests.get(API_BASE, timeout=10)

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Connection successful!")
        print(f"⏱️  Response time: {duration:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")

        results["network_test"] = {
            "success": True,
            "duration": duration,
            "status_code": response.status_code
        }

    except requests.exceptions.Timeout:
        print("❌ Connection TIMEOUT (10 seconds)")
        print("   This indicates network issues or endpoint is very slow")
        results["network_test"] = {
            "success": False,
            "error": "Timeout"
        }

    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection FAILED: {str(e)}")
        print("   Endpoint may be unreachable or URL is incorrect")
        results["network_test"] = {
            "success": False,
            "error": str(e)
        }

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        results["network_test"] = {
            "success": False,
            "error": str(e)
        }


def test_llm_endpoint():
    """Test LLM endpoint with gpt-5-nano model"""
    print("\n" + "="*70)
    print("TEST 2: LLM Endpoint (gpt-5-nano)")
    print("="*70)

    if not API_KEY:
        print("❌ SKIPPED: API_KEY not found")
        return

    try:
        print("🤖 Testing LLM endpoint...")
        print(f"   Model: gpt-5-nano")
        print(f"   Endpoint: {API_BASE}/chat/completions")
        print(f"   Timeout: 30 seconds")

        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        payload = {
            "model": "gpt-5-nano",
            "messages": [
                {"role": "user", "content": "Say 'test' if you can hear me."}
            ],
            "max_tokens": 10
        }

        response = requests.post(
            f"{API_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        end_time = time.time()
        duration = end_time - start_time

        if response.status_code == 200:
            data = response.json()
            print(f"✅ LLM endpoint is working!")
            print(f"⏱️  Response time: {duration:.2f} seconds")
            print(f"📝 Response: {data.get('choices', [{}])[0].get('message', {}).get('content', 'N/A')}")

            results["llm_test"] = {
                "success": True,
                "duration": duration,
                "model_valid": True
            }

        else:
            print(f"❌ LLM request FAILED")
            print(f"📊 Status code: {response.status_code}")
            print(f"📝 Response: {response.text}")

            # Check if it's a model-not-found error
            if "model" in response.text.lower() or "not found" in response.text.lower():
                print("\n⚠️  LIKELY ISSUE: Model 'gpt-5-nano' may not be valid!")
                print("   Suggested models to try:")
                print("   - gpt-4o-mini")
                print("   - gpt-3.5-turbo")
                print("   - gpt-4")

            results["llm_test"] = {
                "success": False,
                "duration": duration,
                "status_code": response.status_code,
                "error": response.text
            }

    except requests.exceptions.Timeout:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ LLM request TIMEOUT after {duration:.2f} seconds")
        print("   This is likely causing your agent timeouts!")

        results["llm_test"] = {
            "success": False,
            "error": "Timeout",
            "duration": duration
        }

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        results["llm_test"] = {
            "success": False,
            "error": str(e)
        }


def test_embedding_endpoint():
    """Test embedding endpoint with text-embedding-3-small"""
    print("\n" + "="*70)
    print("TEST 3: Embedding Endpoint (text-embedding-3-small)")
    print("="*70)

    if not EMBEDDING_KEY:
        print("❌ SKIPPED: EMBEDDING_KEY not found")
        return

    try:
        print("🔢 Testing embedding endpoint...")
        print(f"   Model: text-embedding-3-small")
        print(f"   Endpoint: {API_BASE}/embeddings")
        print(f"   Timeout: 30 seconds")

        start_time = time.time()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {EMBEDDING_KEY}"
        }

        payload = {
            "model": "text-embedding-3-small",
            "input": "test embedding"
        }

        response = requests.post(
            f"{API_BASE}/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )

        end_time = time.time()
        duration = end_time - start_time

        if response.status_code == 200:
            data = response.json()
            embedding_length = len(data.get('data', [{}])[0].get('embedding', []))
            print(f"✅ Embedding endpoint is working!")
            print(f"⏱️  Response time: {duration:.2f} seconds")
            print(f"📊 Embedding dimension: {embedding_length}")

            results["embedding_test"] = {
                "success": True,
                "duration": duration,
                "embedding_dim": embedding_length
            }

        else:
            print(f"❌ Embedding request FAILED")
            print(f"📊 Status code: {response.status_code}")
            print(f"📝 Response: {response.text}")

            results["embedding_test"] = {
                "success": False,
                "duration": duration,
                "status_code": response.status_code,
                "error": response.text
            }

    except requests.exceptions.Timeout:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Embedding request TIMEOUT after {duration:.2f} seconds")
        print("   This could cause timeouts during query embedding!")

        results["embedding_test"] = {
            "success": False,
            "error": "Timeout",
            "duration": duration
        }

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        results["embedding_test"] = {
            "success": False,
            "error": str(e)
        }


def print_summary():
    """Print summary of all tests"""
    print("\n" + "="*70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*70)

    # Network Test
    network = results.get("network_test", {})
    if network:
        print("\n1. Network Connectivity:")
        if network.get("success"):
            print(f"   ✅ PASS ({network.get('duration', 0):.2f}s)")
        else:
            print(f"   ❌ FAIL - {network.get('error', 'Unknown')}")

    # LLM Test
    llm = results.get("llm_test", {})
    if llm:
        print("\n2. LLM Endpoint (gpt-5-nano):")
        if llm.get("success"):
            print(f"   ✅ PASS ({llm.get('duration', 0):.2f}s)")
        else:
            error = llm.get("error", "Unknown")
            print(f"   ❌ FAIL - {error}")
            if "Timeout" in error:
                print("   🚨 THIS IS LIKELY CAUSING YOUR AGENT TIMEOUTS!")

    # Embedding Test
    embedding = results.get("embedding_test", {})
    if embedding:
        print("\n3. Embedding Endpoint:")
        if embedding.get("success"):
            print(f"   ✅ PASS ({embedding.get('duration', 0):.2f}s)")
        else:
            error = embedding.get("error", "Unknown")
            print(f"   ❌ FAIL - {error}")

    # Recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)

    if not results.get("network_test", {}).get("success"):
        print("\n❌ Network connectivity failed:")
        print("   1. Check your internet connection")
        print("   2. Verify API_BASE URL is correct")
        print("   3. Check if firewall is blocking the endpoint")

    if results.get("llm_test", {}).get("error") == "Timeout":
        print("\n🚨 LLM endpoint is timing out:")
        print("   1. Increase timeout from 80s to 120s or 180s")
        print("   2. Check if endpoint has rate limiting")
        print("   3. Consider using official OpenAI endpoint")
        print("   4. Test at different times to see if it's congestion")

    if results.get("llm_test", {}).get("status_code") in [400, 404]:
        print("\n⚠️  Model 'gpt-5-nano' may be invalid:")
        print("   1. Try changing to 'gpt-4o-mini' or 'gpt-3.5-turbo'")
        print("   2. Contact endpoint provider to verify available models")

    if results.get("embedding_test", {}).get("error") == "Timeout":
        print("\n⚠️  Embedding endpoint is timing out:")
        print("   1. Add timeout=60 to OpenAIEmbedding in embedding_loader.py")
        print("   2. This could slow down retrieval phase")

    all_pass = all([
        results.get("network_test", {}).get("success"),
        results.get("llm_test", {}).get("success"),
        results.get("embedding_test", {}).get("success")
    ])

    if all_pass:
        print("\n✅ All tests passed!")
        print("   Your endpoint is working correctly.")
        print("   If you still get timeouts, the issue may be:")
        print("   - Large context causing slow response")
        print("   - Intermittent network issues")
        print("   - Rate limiting kicking in under load")

    print("\n" + "="*70)
    print(f"⏱️  Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == "__main__":
    # Run all tests
    test_network_connectivity()
    test_llm_endpoint()
    test_embedding_endpoint()

    # Print summary
    print_summary()
