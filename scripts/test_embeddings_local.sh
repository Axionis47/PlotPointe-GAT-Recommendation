#!/bin/bash
# Local testing script for embeddings pipeline
# Tests all components before submitting to Vertex AI

set -e

PROJECT_ID="${1:-plotpointe}"
REGION="${2:-us-central1}"

echo "=========================================="
echo "EMBEDDINGS PIPELINE - LOCAL TEST"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass_test() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

fail_test() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    exit 1
}

warn_test() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

echo "=========================================="
echo "TEST 1: Check Python dependencies"
echo "=========================================="

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    pass_test "Python 3 is installed ($(python3 --version))"
else
    fail_test "Python 3 is not installed"
fi

# Check required packages
echo "Checking Python packages..."
python3 -c "import sentence_transformers" 2>/dev/null && pass_test "sentence-transformers installed" || warn_test "sentence-transformers not installed (will be installed in container)"
python3 -c "import transformers" 2>/dev/null && pass_test "transformers installed" || warn_test "transformers not installed (will be installed in container)"
python3 -c "import pandas" 2>/dev/null && pass_test "pandas installed" || warn_test "pandas not installed (will be installed in container)"
python3 -c "import torch" 2>/dev/null && pass_test "torch installed" || warn_test "torch not installed (will be installed in container)"

echo ""
echo "=========================================="
echo "TEST 2: Check GCS permissions"
echo "=========================================="

# Check if gcloud is configured
if ! gcloud config get-value project &> /dev/null; then
    fail_test "gcloud is not configured"
fi
pass_test "gcloud is configured"

# Check service account permissions
SA_EMAIL="sa-pipeline@${PROJECT_ID}.iam.gserviceaccount.com"
echo "Checking service account: $SA_EMAIL"

# Get service account roles
SA_ROLES=$(gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:$SA_EMAIL" \
    --format="value(bindings.role)" 2>/dev/null || echo "")

if [ -z "$SA_ROLES" ]; then
    fail_test "Service account $SA_EMAIL has no project-level roles"
fi

echo "Service account roles:"
echo "$SA_ROLES" | while read role; do
    echo "  - $role"
done

# Check for required roles
if echo "$SA_ROLES" | grep -q "roles/storage"; then
    pass_test "Service account has storage role"
elif echo "$SA_ROLES" | grep -q "roles/editor"; then
    pass_test "Service account has editor role (includes storage)"
else
    fail_test "Service account missing storage permissions (needs roles/storage.objectAdmin or roles/storage.objectUser)"
fi

echo ""
echo "=========================================="
echo "TEST 3: Check GCS bucket access"
echo "=========================================="

# Test bucket read access
BUCKET="gs://plotpointe-artifacts"
echo "Testing bucket: $BUCKET"

if gsutil ls $BUCKET > /dev/null 2>&1; then
    pass_test "Can list bucket contents"
else
    fail_test "Cannot list bucket contents"
fi

# Test staging data access
STAGING_PATH="$BUCKET/staging/amazon_electronics"
echo "Testing staging data: $STAGING_PATH"

if gsutil ls $STAGING_PATH/items.parquet > /dev/null 2>&1; then
    pass_test "items.parquet exists"
else
    fail_test "items.parquet not found at $STAGING_PATH"
fi

if gsutil ls $STAGING_PATH/interactions.parquet > /dev/null 2>&1; then
    pass_test "interactions.parquet exists"
else
    fail_test "interactions.parquet not found at $STAGING_PATH"
fi

# Test code upload location
CODE_PATH="$BUCKET/code/embeddings"
echo "Testing code location: $CODE_PATH"

if gsutil ls $CODE_PATH/ > /dev/null 2>&1; then
    pass_test "Code directory exists"
    gsutil ls $CODE_PATH/*.py 2>/dev/null | while read file; do
        echo "  - $(basename $file)"
    done
else
    warn_test "Code directory doesn't exist yet (will be created)"
fi

# Test write access
echo "Testing write access..."
TEST_FILE="$BUCKET/test_write_$(date +%s).txt"
if echo "test" | gsutil cp - $TEST_FILE 2>/dev/null; then
    pass_test "Can write to bucket"
    gsutil rm $TEST_FILE 2>/dev/null
else
    fail_test "Cannot write to bucket"
fi

echo ""
echo "=========================================="
echo "TEST 4: Check Vertex AI setup"
echo "=========================================="

# Check if Vertex AI API is enabled
if gcloud services list --enabled --filter="name:aiplatform.googleapis.com" --format="value(name)" | grep -q aiplatform; then
    pass_test "Vertex AI API is enabled"
else
    fail_test "Vertex AI API is not enabled"
fi

# Check if experiment exists
EXPERIMENT_NAME="recsys-dev"
echo "Checking experiment: $EXPERIMENT_NAME"

if gcloud ai experiments describe $EXPERIMENT_NAME --region=$REGION --project=$PROJECT_ID > /dev/null 2>&1; then
    pass_test "Experiment '$EXPERIMENT_NAME' exists"
else
    warn_test "Experiment '$EXPERIMENT_NAME' doesn't exist (will be created on first run)"
fi

echo ""
echo "=========================================="
echo "TEST 5: Validate Python scripts"
echo "=========================================="

# Check if scripts exist
for script in embed_text.py embed_image.py fuse_modal.py; do
    if [ -f "embeddings/$script" ]; then
        pass_test "embeddings/$script exists"
        
        # Check for syntax errors
        if python3 -m py_compile "embeddings/$script" 2>/dev/null; then
            pass_test "embeddings/$script has valid syntax"
        else
            fail_test "embeddings/$script has syntax errors"
        fi
    else
        fail_test "embeddings/$script not found"
    fi
done

echo ""
echo "=========================================="
echo "TEST 6: Validate Vertex AI configs"
echo "=========================================="

# Check if config files exist
for config in embed_text.yaml embed_image.yaml fuse_modal.yaml; do
    if [ -f "vertex/configs/$config" ]; then
        pass_test "vertex/configs/$config exists"
        
        # Basic YAML validation
        if python3 -c "import yaml; yaml.safe_load(open('vertex/configs/$config'))" 2>/dev/null; then
            pass_test "vertex/configs/$config is valid YAML"
        else
            fail_test "vertex/configs/$config has YAML syntax errors"
        fi
        
        # Check for required fields
        if grep -q "serviceAccount:" "vertex/configs/$config"; then
            pass_test "vertex/configs/$config has serviceAccount field"
        else
            fail_test "vertex/configs/$config missing serviceAccount field"
        fi
        
        # Check if service account is correct
        if grep -q "$SA_EMAIL" "vertex/configs/$config"; then
            pass_test "vertex/configs/$config uses correct service account"
        else
            fail_test "vertex/configs/$config uses wrong service account"
        fi
    else
        fail_test "vertex/configs/$config not found"
    fi
done

echo ""
echo "=========================================="
echo "TEST 7: Test data download (small sample)"
echo "=========================================="

# Download a small sample to test access
TMP_DIR=$(mktemp -d)
echo "Downloading sample data to $TMP_DIR..."

if gsutil cp $STAGING_PATH/items.parquet $TMP_DIR/items.parquet 2>/dev/null; then
    pass_test "Successfully downloaded items.parquet"
    
    # Check file size
    FILE_SIZE=$(du -h $TMP_DIR/items.parquet | cut -f1)
    echo "  File size: $FILE_SIZE"
    
    # Try to read with pandas
    if python3 -c "import pandas as pd; df = pd.read_parquet('$TMP_DIR/items.parquet'); print(f'Rows: {len(df)}, Columns: {len(df.columns)}')" 2>/dev/null; then
        pass_test "Successfully read parquet file with pandas"
    else
        fail_test "Cannot read parquet file with pandas"
    fi
else
    fail_test "Cannot download items.parquet"
fi

# Cleanup
rm -rf $TMP_DIR

echo ""
echo "=========================================="
echo "TEST 8: Check container images"
echo "=========================================="

# Check if Deep Learning Container images are accessible
DLC_CPU="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cpu.py310"
DLC_GPU="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu121.2-1.py310"

echo "Checking container images..."
echo "  CPU: $DLC_CPU"
echo "  GPU: $DLC_GPU"

# Note: We can't easily test image pull without Docker, so just check if they're referenced correctly
if grep -q "$DLC_CPU" vertex/configs/embed_text.yaml; then
    pass_test "embed_text.yaml uses correct CPU container"
else
    fail_test "embed_text.yaml uses wrong container image"
fi

if grep -q "$DLC_GPU" vertex/configs/embed_image.yaml; then
    pass_test "embed_image.yaml uses correct GPU container"
else
    fail_test "embed_image.yaml uses wrong container image"
fi

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo -e "${GREEN}All tests passed!${NC}"
echo ""
echo "Next steps:"
echo "1. Fix service account permissions if needed:"
echo "   gcloud projects add-iam-policy-binding $PROJECT_ID \\"
echo "     --member='serviceAccount:$SA_EMAIL' \\"
echo "     --role='roles/storage.objectAdmin'"
echo ""
echo "2. Run the embeddings pipeline:"
echo "   bash scripts/run_embeddings_pipeline.sh $PROJECT_ID $REGION"
echo ""

