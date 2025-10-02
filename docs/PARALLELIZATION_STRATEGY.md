# Parallelization Strategy

## 📊 Available Compute Resources

### Current Quotas (us-central1)
| Resource | Limit | In Use | Available |
|----------|-------|--------|-----------|
| **CPUs** | 200 | 0 | **200** ✅ |
| **T4 GPUs (on-demand)** | 1 | 1 | **0** ⚠️ |
| **T4 GPUs (preemptible)** | 1 | 0 | **1** ✅ |
| **L4 GPUs (on-demand)** | 1 | 0 | **1** ✅ |
| **L4 GPUs (preemptible)** | 1 | 0 | **1** ✅ |

### GPU Comparison
| GPU | Memory | Performance | Cost (on-demand) | Best For |
|-----|--------|-------------|------------------|----------|
| **T4** | 16 GB | 1x | $0.35/hour | Standard workloads |
| **L4** | 24 GB | **2x faster** | $0.70/hour | Faster inference, larger batches |

**Key Insight:** L4 is 2x faster but only 2x cost → **Same cost per computation, but 2x faster completion!**

---

## 🎯 Optimization Strategy

### Current Situation
- **Current job:** T4 GPU, batch_size=32, ~7-8 hours
- **Problem:** Slow throughput (5.4 items/sec)
- **Opportunity:** Can optimize GPU + parallelize CPU tasks

### Recommended Approach: **Optimize + Parallelize**

#### Track 1: Optimized Image Embeddings (GPU)
- **Cancel current T4 job** (only 15 min in)
- **Switch to L4 GPU** (2x faster)
- **Increase batch_size** to 64 (better GPU utilization)
- **Expected time:** 3-4 hours (vs 7-8 hours)
- **Savings:** 4-5 hours!

#### Track 2: Parallel CPU Tasks (Start NOW)
While image embeddings run, start:

1. **U-I Edge Construction** (10-15 min)
   - Uses `interactions.parquet` (already available)
   - Builds user-item bipartite graph
   - Output: `ui_edges.npz`

2. **Text-based I-I kNN** (15-20 min)
   - Uses `txt.npy` (already complete!)
   - Builds item-item similarity graph
   - Output: `ii_edges_txt.npz`

3. **Baseline MF/BPR Training** (30-45 min)
   - CPU-only matrix factorization
   - Establishes baseline metrics
   - Output: Model + eval.json

---

## ⏱️ Timeline Comparison

### Sequential (Current Approach)
```
Image Embeddings (T4):     [========================================] 7-8 hours
Modal Fusion:                                                        [==] 20 min
U-I Edges:                                                              [=] 15 min
Text I-I kNN:                                                            [=] 20 min
Fused I-I kNN:                                                             [=] 20 min
────────────────────────────────────────────────────────────────────────────────
Total:                                                                   ~9 hours
```

### Parallel (Optimized Approach)
```
Image Embeddings (L4):     [====================] 3-4 hours
Modal Fusion:                                    [=] 20 min
U-I Edges:                 [=] 15 min
Text I-I kNN:              [=] 20 min
Fused I-I kNN:                                      [=] 20 min
────────────────────────────────────────────────────────────────────────────────
Total:                                              ~4.5 hours
```

**Time Savings: 4.5 hours (50% faster!)**

---

## 💰 Cost Analysis

### Sequential Approach
- T4 GPU: 8 hours × $0.35/hour = $2.80
- CPU tasks: 1.5 hours × $0.10/hour = $0.15
- **Total: $2.95**

### Parallel Approach
- L4 GPU: 4 hours × $0.70/hour = $2.80
- CPU tasks (parallel): 0.5 hours × $0.30/hour = $0.15
- **Total: $2.95**

**Cost: Same! But 4.5 hours faster!**

---

## 🚀 Execution Plan

### Step 1: Upload Latest Code
```bash
gsutil cp embeddings/embed_image.py gs://plotpointe-artifacts/code/embeddings/embed_image.py
```

### Step 2: Cancel Current Job (if needed)
```bash
gcloud ai custom-jobs cancel projects/359145045403/locations/us-central1/customJobs/1860108365477183488 --region=us-central1
```

### Step 3: Launch Parallel Pipeline
```bash
bash scripts/parallel_pipeline.sh plotpointe us-central1
```

This will:
1. ✅ Cancel current T4 job
2. ✅ Launch L4 job (image embeddings)
3. ✅ Launch CPU job (U-I edges) - if script exists
4. ✅ Launch CPU job (text I-I kNN) - if script exists

---

## 📋 What Gets Parallelized

### Can Run in Parallel (Independent)
- ✅ **Image embeddings** (needs: items.parquet)
- ✅ **U-I edges** (needs: interactions.parquet)
- ✅ **Text I-I kNN** (needs: txt.npy) ← **Already available!**
- ✅ **Baseline MF/BPR** (needs: interactions.parquet)

### Must Run Sequentially (Dependencies)
- ⏳ **Modal fusion** (needs: txt.npy + img.npy)
- ⏳ **Fused I-I kNN** (needs: fused.npy)
- ⏳ **GAT training** (needs: all graphs + embeddings)

---

## 🎮 Future: GPU Quota Increase

When your GPU quota increase is approved, you can parallelize GAT training:

### GAT Training Experiments (4 GPUs)
```
GPU 1: GAT (text-only, 1 head, 2 layers)
GPU 2: GAT (text-only, 4 heads, 2 layers)
GPU 3: GAT (fused, 4 heads, 2 layers)
GPU 4: GAT (fused, 8 heads, 3 layers)
```

**Sequential:** 4 experiments × 1.5 hours = 6 hours  
**Parallel:** 1.5 hours (4x speedup!)

---

## 📊 Resource Utilization

### Current (Sequential)
- GPU utilization: 12.5% (1 GPU for 8 hours out of 64 GPU-hours available)
- CPU utilization: ~5% (1 CPU job at a time)

### Optimized (Parallel)
- GPU utilization: 25% (1 L4 for 4 hours)
- CPU utilization: ~15% (3 parallel CPU jobs)
- **Better resource utilization + faster completion!**

---

## 🔔 Next Steps After Image Embeddings

### Immediate (20-40 min)
1. Modal fusion (CPU, 20 min)
2. Fused I-I kNN (CPU, 20 min)

### Short-term (1-2 hours)
3. Baseline models (MF/BPR, LightGCN)
4. Evaluation metrics setup

### Medium-term (3-5 hours)
5. GAT training (single GPU)
6. Ablation studies

### With GPU Quota (1-2 hours)
7. Parallel GAT experiments (4 GPUs)
8. Hyperparameter tuning

---

## 📈 Monitoring

### Check Running Jobs
```bash
gcloud ai custom-jobs list --region=us-central1 --filter='state:JOB_STATE_RUNNING'
```

### Monitor Specific Job
```bash
JOB_ID="<your-job-id>"
gcloud ai custom-jobs describe projects/359145045403/locations/us-central1/customJobs/$JOB_ID --region=us-central1
```

### View Logs
```bash
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=$JOB_ID" --limit=50 --project=plotpointe
```

---

## ✅ Summary

### Recommended Action: **Launch Parallel Pipeline**

**Command:**
```bash
bash scripts/parallel_pipeline.sh plotpointe us-central1
```

**Benefits:**
- ✅ 4.5 hours faster (50% time savings)
- ✅ Same cost ($2.95)
- ✅ Better resource utilization
- ✅ Parallel CPU tasks start immediately
- ✅ L4 GPU is 2x faster than T4

**Expected Completion:**
- CPU tasks: 15-20 minutes
- Image embeddings: 3-4 hours
- Modal fusion: +20 minutes
- **Total: ~4.5 hours** (vs 9 hours sequential)

**Next Steps:**
1. Run the parallel pipeline script
2. Monitor progress (Terminal 68 or console)
3. After completion, proceed to GAT training
4. Utilize GPU quota increase for parallel experiments

