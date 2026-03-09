#!/bin/bash
# Quick Start Script for RecRL

echo "🚀 RecRL Quick Start"
echo "===================="
echo ""

# Check if we're in the right directory
if [ ! -d "recrl" ]; then
    echo "❌ Error: recrl directory not found. Please run this script from CleanOneRec/"
    exit 1
fi

echo "📦 Step 1: Installing RecRL..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "   Using uv package manager..."
    uv pip install -e recrl/
    if [ $? -eq 0 ]; then
        echo "✅ RecRL installed successfully with uv"
    else
        echo "❌ Installation failed"
        exit 1
    fi
else
    echo "   Using pip package manager..."
    pip install -e recrl/ -q
    if [ $? -eq 0 ]; then
        echo "✅ RecRL installed successfully with pip"
    else
        echo "❌ Installation failed"
        exit 1
    fi
fi

echo ""
echo "📚 Step 2: Available Examples"
echo "   - recrl/examples/train_grpo.py       (GRPO training)"
echo "   - recrl/examples/train_eepo.py       (EEPO training)"
echo "   - recrl/examples/train_composite.py  (Composite rewards)"

echo ""
echo "📖 Step 3: Documentation"
echo "   - ARCHITECTURE.md  (Design principles)"
echo "   - MIGRATION.md     (Migrate from MiniOneRec)"
echo "   - SUMMARY.md       (Framework overview)"
echo "   - recrl/README.md  (User guide)"

echo ""
echo "🎯 Quick Test"
echo "Run this to verify installation:"
echo ""
echo "  python -c 'from recrl.core import DataEngine, RolloutEngine; print(\"✅ RecRL working!\")'"
echo ""

python -c 'from recrl.core import DataEngine, RolloutEngine; print("✅ RecRL is working!")'

echo ""
echo "🎉 Setup complete! You can now:"
echo "   1. Check examples: cd recrl/examples/"
echo "   2. Read docs: cat ARCHITECTURE.md"
echo "   3. Start training: python recrl/examples/train_grpo.py"
echo ""
