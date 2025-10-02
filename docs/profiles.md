# Vertex training profiles

## gpu-t4-preemptible
- Single GPU (NVIDIA T4), cost-optimized Spot VMs (preemptible)
- Machine: n1-standard-8 + 1x T4
- Scheduling: Spot strategy with restart on worker restart
- Config: see `vertex/profiles/gpu-t4-preemptible.yaml`

When to use:
- Baseline training, light to medium compute workloads
- Cost-sensitive experimentation where preemption risk is acceptable

When to pick L4 instead:
- You need better performance/efficiency for modern DL workloads (transformers, CV)
- Availability of G2 (L4) in your target zones is good (e.g., us-central1 a/b/c)
- Use machineType `g2-standard-8` and `acceleratorType: NVIDIA_L4`

Notes:
- For non-preemptible runs, remove `scheduling.strategy: SPOT` from the profile.
- Consider enabling periodic checkpointing to tolerate Spot preemptions.

