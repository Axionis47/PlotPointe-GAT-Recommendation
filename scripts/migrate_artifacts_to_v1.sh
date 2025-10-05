#!/bin/bash
# Migrate existing artifacts to v1/ (baseline version)

set -e

echo "🔄 Migrating artifacts to v1/ (baseline)"
echo "=========================================="
echo ""

# Create v1 directories
echo "📁 Creating v1 directories..."
gsutil -m mkdir gs://plotpointe-artifacts/embeddings/v1/ 2>/dev/null || true
gsutil -m mkdir gs://plotpointe-artifacts/graphs/v1/ 2>/dev/null || true
echo "✅ Directories created"
echo ""

# Move embeddings to v1
echo "📦 Moving embeddings to v1/..."
gsutil -m mv gs://plotpointe-artifacts/embeddings/txt.npy \
              gs://plotpointe-artifacts/embeddings/v1/ 2>/dev/null || echo "  ⚠️  txt.npy already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/embeddings/txt_meta.json \
              gs://plotpointe-artifacts/embeddings/v1/ 2>/dev/null || echo "  ⚠️  txt_meta.json already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/embeddings/img.npy \
              gs://plotpointe-artifacts/embeddings/v1/ 2>/dev/null || echo "  ⚠️  img.npy already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/embeddings/img_items.parquet \
              gs://plotpointe-artifacts/embeddings/v1/ 2>/dev/null || echo "  ⚠️  img_items.parquet already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/embeddings/img_meta.json \
              gs://plotpointe-artifacts/embeddings/v1/ 2>/dev/null || echo "  ⚠️  img_meta.json already moved or doesn't exist"
echo "✅ Embeddings migrated"
echo ""

# Move graphs to v1
echo "📊 Moving graphs to v1/..."
gsutil -m mv gs://plotpointe-artifacts/graphs/ui_edges.npz \
              gs://plotpointe-artifacts/graphs/v1/ 2>/dev/null || echo "  ⚠️  ui_edges.npz already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/graphs/ui_stats.json \
              gs://plotpointe-artifacts/graphs/v1/ 2>/dev/null || echo "  ⚠️  ui_stats.json already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/graphs/ii_edges_txt.npz \
              gs://plotpointe-artifacts/graphs/v1/ 2>/dev/null || echo "  ⚠️  ii_edges_txt.npz already moved or doesn't exist"
gsutil -m mv gs://plotpointe-artifacts/graphs/ii_edges_txt_stats.json \
              gs://plotpointe-artifacts/graphs/v1/ 2>/dev/null || echo "  ⚠️  ii_edges_txt_stats.json already moved or doesn't exist"
echo "✅ Graphs migrated"
echo ""

# Verify migration
echo "🔍 Verifying migration..."
echo ""
echo "v1 embeddings:"
gsutil ls -lh gs://plotpointe-artifacts/embeddings/v1/
echo ""
echo "v1 graphs:"
gsutil ls -lh gs://plotpointe-artifacts/graphs/v1/
echo ""

echo "✅ Migration complete!"
echo ""
echo "📂 Old artifacts preserved in:"
echo "   gs://plotpointe-artifacts/embeddings/v1/"
echo "   gs://plotpointe-artifacts/graphs/v1/"
echo ""
echo "📂 New artifacts will be generated in:"
echo "   gs://plotpointe-artifacts/embeddings/v2/"
echo "   gs://plotpointe-artifacts/graphs/v2/"
echo ""
echo "💡 Next steps:"
echo "   1. Update Vertex AI configs to use v2/ paths"
echo "   2. Run improved pipeline"
echo "   3. Compare v1 vs v2 performance"

