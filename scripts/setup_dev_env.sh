#!/bin/bash
# ============================================================================
# PlotPointe - Development Environment Setup
# ============================================================================
# This script sets up a Python 3.10 virtual environment with all dependencies
# Compatible with GCP Vertex AI containers (py310)
# ============================================================================

set -e  # Exit on error

echo "🚀 PlotPointe Development Environment Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "   Found: Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -ne 3 ] || [ "$PYTHON_MINOR" -lt 10 ] || [ "$PYTHON_MINOR" -gt 12 ]; then
    echo -e "${RED}❌ Error: Python 3.10, 3.11, or 3.12 required${NC}"
    echo "   Current version: $PYTHON_VERSION"
    echo ""
    echo "   Install Python 3.10 using:"
    echo "   - macOS: brew install python@3.10"
    echo "   - Ubuntu: sudo apt install python3.10 python3.10-venv"
    exit 1
fi

if [ "$PYTHON_MINOR" -eq 12 ]; then
    echo -e "${YELLOW}⚠️  Warning: You're using Python 3.12${NC}"
    echo "   GCP Vertex AI uses Python 3.10"
    echo "   Consider using Python 3.10 for full compatibility"
    echo ""
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}✅ Python version compatible${NC}"
echo ""

# Remove old virtual environment if exists
if [ -d ".venv" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf .venv
    echo -e "${GREEN}✅ Old environment removed${NC}"
    echo ""
fi

# Create new virtual environment
echo "📦 Creating new virtual environment..."
python3 -m venv .venv
echo -e "${GREEN}✅ Virtual environment created${NC}"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"
echo ""

# Upgrade pip, setuptools, wheel
echo "⬆️  Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✅ Core tools upgraded${NC}"
echo ""

# Install production dependencies
echo "📥 Installing production dependencies..."
echo "   This may take 5-10 minutes (downloading PyTorch, transformers, etc.)"
pip install -r requirements.txt
echo -e "${GREEN}✅ Production dependencies installed${NC}"
echo ""

# Install development dependencies
echo "📥 Installing development dependencies..."
pip install -r requirements-dev.txt
echo -e "${GREEN}✅ Development dependencies installed${NC}"
echo ""

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
echo -e "${GREEN}✅ Pre-commit hooks installed${NC}"
echo ""

# Verify critical packages
echo "🔍 Verifying critical packages..."
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'   Transformers: {transformers.__version__}')"
python3 -c "import google.cloud.aiplatform; print(f'   Vertex AI SDK: {google.cloud.aiplatform.__version__}')"
python3 -c "import pandas; print(f'   Pandas: {pandas.__version__}')"
python3 -c "import pytest; print(f'   Pytest: {pytest.__version__}')"
echo -e "${GREEN}✅ All critical packages verified${NC}"
echo ""

# Check GPU availability
echo "🎮 Checking GPU availability..."
python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'   GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"
echo ""

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p tmp
mkdir -p tests
mkdir -p htmlcov
mkdir -p .pytest_cache
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Summary
echo "=============================================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Run tests:"
echo "     pytest"
echo ""
echo "  3. Check code quality:"
echo "     black --check ."
echo "     flake8 ."
echo "     mypy embeddings/ graphs/"
echo ""
echo "  4. Format code:"
echo "     black ."
echo "     isort ."
echo ""
echo "  5. Run pre-commit checks:"
echo "     pre-commit run --all-files"
echo ""
echo "Happy coding! 🚀"

