# Problem 2 - Compute Selection

## Workload A: 100,000-row sklearn

Selected compute: CPU
Reason:
- Traditional sklearn models are CPU-friendly.
- Dataset size is manageable.
- GPU would add unnecessary cost and complexity.

## Workload B: CNN image classification

Selected compute: GPU
Reason:
- CNNs perform large tensor operations.
- GPU parallel processing can accelerate training.
- Deep learning workloads benefit from GPU hardware.

## Local vs Managed

Local:
- Development
- Debugging
- Small datasets
- Quick experiments

Managed:
- Reproducible cloud jobs
- Larger datasets
- Team collaboration
- Scalable workloads