#!/bin/bash
# Monitor embeddings pipeline progress

PROJECT_ID="${1:-plotpointe}"
REGION="${2:-us-central1}"
JOB_ID="${3}"

if [ -z "$JOB_ID" ]; then
    echo "Usage: $0 <project_id> <region> <job_id>"
    echo "Example: $0 plotpointe us-central1 4062905144935710720"
    exit 1
fi

FULL_JOB_ID="projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/$REGION/customJobs/$JOB_ID"

echo "=========================================="
echo "EMBEDDINGS PIPELINE - STATUS MONITOR"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Job ID: $JOB_ID"
echo ""

# Function to get job state
get_state() {
    gcloud ai custom-jobs describe $FULL_JOB_ID \
        --region=$REGION \
        --format="value(state)" 2>/dev/null
}

# Function to get latest logs
get_latest_logs() {
    gcloud ai custom-jobs stream-logs $FULL_JOB_ID \
        --region=$REGION 2>&1 | tail -20
}

# Function to check outputs
check_outputs() {
    echo ""
    echo "Checking GCS outputs..."
    gsutil ls -lh gs://plotpointe-artifacts/embeddings/ 2>/dev/null || echo "  No outputs yet"
}

# Monitor loop
while true; do
    clear
    echo "=========================================="
    echo "EMBEDDINGS PIPELINE - STATUS MONITOR"
    echo "=========================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Job ID: $JOB_ID"
    echo ""
    
    STATE=$(get_state)
    echo "Status: $STATE"
    echo ""
    
    case "$STATE" in
        "JOB_STATE_SUCCEEDED")
            echo "✅ JOB COMPLETED SUCCESSFULLY!"
            check_outputs
            echo ""
            echo "View results:"
            echo "  gsutil ls -lh gs://plotpointe-artifacts/embeddings/"
            exit 0
            ;;
        "JOB_STATE_FAILED")
            echo "❌ JOB FAILED"
            echo ""
            echo "Latest logs:"
            get_latest_logs
            exit 1
            ;;
        "JOB_STATE_CANCELLED")
            echo "⚠️  JOB CANCELLED"
            exit 1
            ;;
        "JOB_STATE_RUNNING")
            echo "⏳ Job is running..."
            echo ""
            echo "Latest logs:"
            get_latest_logs
            ;;
        *)
            echo "⏸️  Job state: $STATE"
            ;;
    esac
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 30 seconds..."
    sleep 30
done

