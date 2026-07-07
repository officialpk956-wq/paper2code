#!/bin/bash
# Smoke tests for Paper2Code production deployment

set -e

API_URL="${1:-https://api.paper2code.com}"
JWT_TOKEN="${JWT_TOKEN:-test-token}"

echo "🧪 Running smoke tests against $API_URL"

# Test 1: Health check
echo "✓ Health endpoint..."
curl -f -s "$API_URL/health" > /dev/null || { echo "❌ Health check failed"; exit 1; }

# Test 2: Get problems
echo "✓ GET /api/problems..."
curl -f -s "$API_URL/api/problems" | grep -q "id" || { echo "❌ Problems endpoint failed"; exit 1; }

# Test 3: Get papers
echo "✓ GET /api/papers..."
curl -f -s "$API_URL/api/papers" | grep -q "title" || { echo "❌ Papers endpoint failed"; exit 1; }

# Test 4: Get metrics
echo "✓ GET /metrics..."
curl -f -s "$API_URL/metrics" | grep -q "http_requests" || { echo "❌ Metrics endpoint failed"; exit 1; }

# Test 5: Swagger docs
echo "✓ GET /docs..."
curl -f -s "$API_URL/docs" | grep -q "swagger" || { echo "❌ Swagger docs failed"; exit 1; }

# Test 6: Analytics dashboard (anonymous)
echo "✓ GET /api/analytics/dashboard..."
curl -f -s "$API_URL/api/analytics/dashboard" \
  -H "X-Learner-ID: test-user" | grep -q "overview" || { echo "❌ Analytics failed"; exit 1; }

# Test 7: Response time check
echo "✓ Latency check..."
LATENCY=$(curl -w "%{time_total}" -o /dev/null -s "$API_URL/api/problems")
if (( $(echo "$LATENCY > 1.0" | bc -l) )); then
  echo "⚠️  Warning: API latency is ${LATENCY}s (target: < 1s)"
else
  echo "  API latency: ${LATENCY}s ✓"
fi

echo ""
echo "✅ All smoke tests passed!"
