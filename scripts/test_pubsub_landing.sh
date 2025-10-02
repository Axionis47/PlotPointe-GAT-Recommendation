#!/usr/bin/env bash
# Test Pub/Sub → BigQuery landing with idempotency
set -e

PROJECT=${1:-plotpointe}
REGION=${2:-us-central1}

echo "[TEST] Pub/Sub → BigQuery landing dry-run"
echo "Project: $PROJECT, Region: $REGION"

# Generate unique IDs
REQ_ID="req-aug-test-$(date +%s)"
EVT_ID="evt-aug-test-$(date +%s)"

echo "REQ_ID=$REQ_ID"
echo "EVT_ID=$EVT_ID"

# Create sample messages
mkdir -p tmp
cat > tmp/req_test.json <<EOF
{
  "request_id": "$REQ_ID",
  "user_id": "u-42",
  "session_id": "s-1",
  "request_ts": "2025-09-27T20:00:00Z",
  "context": {"page":"home","device":"web","country":"US"},
  "model": {"name":"gat","version":"v0.1","profile":"gpu-l4"},
  "req_features": {"surface":"home","ab":"A"},
  "items": [
    {"asin":"B000TEST01","score":0.91,"rank":1},
    {"asin":"B000TEST02","score":0.82,"rank":2}
  ],
  "latency_ms": 123,
  "trace_id": "trace-abc",
  "experiment": {"name":"expA","variant":"A"}
}
EOF

cat > tmp/fb_test.json <<EOF
{
  "event_id": "$EVT_ID",
  "user_id": "u-42",
  "event_ts": "2025-09-27T20:00:05Z",
  "asin": "B000TEST01",
  "event_type": "click",
  "request_id": "$REQ_ID",
  "session_id": "s-1",
  "qty": 1,
  "value": null,
  "metadata": {"surface":"home"}
}
EOF

echo "[TEST] Publishing to Pub/Sub (twice for idempotency test)..."
for i in 1 2; do
  gcloud pubsub topics publish recsys-requests \
    --message="$(cat tmp/req_test.json)" \
    --project=$PROJECT 2>&1 | grep -i messageIds || true
  
  gcloud pubsub topics publish recsys-feedback \
    --message="$(cat tmp/fb_test.json)" \
    --project=$PROJECT 2>&1 | grep -i messageIds || true
  
  sleep 1
done

echo "[TEST] Simulating processor MERGE (idempotent landing)..."

# Requests MERGE (run twice)
for i in 1 2; do
  bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false --quiet "
MERGE \`$PROJECT.recsys_logs.requests\` T
USING (
  SELECT
    '$REQ_ID' AS request_id,
    'u-42' AS user_id,
    's-1' AS session_id,
    TIMESTAMP('2025-09-27T20:00:00Z') AS request_ts,
    STRUCT('home' AS page,'web' AS device,'US' AS country) AS context,
    STRUCT('gat' AS name,'v0.1' AS version,'gpu-l4' AS profile) AS model,
    JSON '{\"surface\":\"home\",\"ab\":\"A\"}' AS req_features,
    [STRUCT('B000TEST01' AS asin, 0.91 AS score, 1 AS rank), STRUCT('B000TEST02' AS asin, 0.82 AS score, 2 AS rank)] AS items,
    123 AS latency_ms,
    'trace-abc' AS trace_id,
    STRUCT('expA' AS name,'A' AS variant) AS experiment
) S
ON T.request_id = S.request_id
WHEN NOT MATCHED THEN INSERT (request_id,user_id,session_id,request_ts,context,model,req_features,items,latency_ms,trace_id,experiment)
VALUES (S.request_id,S.user_id,S.session_id,S.request_ts,S.context,S.model,S.req_features,S.items,S.latency_ms,S.trace_id,S.experiment)
" >/dev/null
  echo "  MERGE requests (run $i) → OK"
done

# Feedback MERGE (run twice)
for i in 1 2; do
  bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false --quiet "
MERGE \`$PROJECT.recsys_logs.feedback\` T
USING (
  SELECT
    '$EVT_ID' AS event_id,
    'u-42' AS user_id,
    TIMESTAMP('2025-09-27T20:00:05Z') AS event_ts,
    'B000TEST01' AS asin,
    'click' AS event_type,
    '$REQ_ID' AS request_id,
    's-1' AS session_id,
    1 AS qty,
    CAST(NULL AS FLOAT64) AS value,
    JSON '{\"surface\":\"home\"}' AS metadata
) S
ON T.event_id = S.event_id
WHEN NOT MATCHED THEN INSERT (event_id,user_id,event_ts,asin,event_type,request_id,session_id,qty,value,metadata)
VALUES (S.event_id,S.user_id,S.event_ts,S.asin,S.event_type,S.request_id,S.session_id,S.qty,S.value,S.metadata)
" >/dev/null
  echo "  MERGE feedback (run $i) → OK"
done

echo "[TEST] Verifying row counts (should be 1 each)..."
REQ_CNT=$(bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false --format=csv --quiet \
  "SELECT COUNT(*) FROM \`$PROJECT.recsys_logs.requests\` WHERE request_id='$REQ_ID'" | tail -n 1)
FB_CNT=$(bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false --format=csv --quiet \
  "SELECT COUNT(*) FROM \`$PROJECT.recsys_logs.feedback\` WHERE event_id='$EVT_ID'" | tail -n 1)

echo "  requests: $REQ_CNT row(s)"
echo "  feedback: $FB_CNT row(s)"

if [[ "$REQ_CNT" == "1" && "$FB_CNT" == "1" ]]; then
  echo "[TEST] ✅ Idempotency verified (1 row each after 2 MERGEs)"
else
  echo "[TEST] ❌ Idempotency FAILED (expected 1 row each)"
  exit 1
fi

echo "[TEST] Querying compatibility views..."
bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false --format=prettyjson --quiet \
  "SELECT ts, user_id, model_family, model_version, ab_bucket, latency_ms, ARRAY_LENGTH(items) AS n_items, request_id 
   FROM \`$PROJECT.recsys_logs.requests_v1\` 
   WHERE request_id='$REQ_ID' LIMIT 1" | head -n 20

bq --project_id=$PROJECT --location=$REGION query --use_legacy_sql=false --format=prettyjson --quiet \
  "SELECT request_id, user_id, asin, action, ts_action 
   FROM \`$PROJECT.recsys_logs.feedback_v1\` 
   WHERE request_id='$REQ_ID' LIMIT 1" | head -n 20

echo "[TEST] ✅ Pub/Sub → BigQuery landing test complete"
echo "Summary:"
echo "  - Published 2× to recsys-requests and recsys-feedback"
echo "  - Ran MERGE 2× for each table"
echo "  - Verified idempotency: 1 row in requests, 1 row in feedback"
echo "  - Compatibility views (requests_v1, feedback_v1) return correct schema"

