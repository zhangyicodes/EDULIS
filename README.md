# EDULIS: Edge-Aware Deep Unfolding Network for Low-Light Instance Segmentation

This project is based on the MMDetection framework.

## Abstract

Instance segmentation in low-light conditions poses serious challenges due to dark noise and compressed dynamic range, leading to object omission and boundary inaccuracies. Existing methods, typically employing pre-processing enhancement or generic multi-scale fusion, often fail to effectively mitigate fundamental degradation mechanisms.
To overcome these limitations, we propose EDULIS, an Edge-Aware Deep Unfolding Network that incorporates Bayesian priors and a deep unfolding strategy to enable principled feature denoising and edge-aware boundary recovery.
Specifically, this approach is realized via two key components: the Unfolding Iterative Feature Pyramid Network (UIFPN),
which implements an unfolded feature optimization algorithm to progressively suppress dark noise while preserving essential structures;
and the Edge-aware Unfolding Segmentation Head (EUSH), which leverages dynamically extracted edge priors through unfolded edge-aware optimization to recover precise object boundaries. 
Experiments on the LIS dataset 
demonstrate that EDULIS outperforms state-of-the-art approaches, validating its effectiveness for robust low-light instance 
segmentation.

## Usage
### Training
#### Single GPU Training:
```python tools/train.py configs/edulis/edulis_r50_fpn_ms_instance.py```
#### Multi-GPU Training:
```./tools/dist_train.sh configs/edulis/edulis_r50_fpn_ms_instance.py 2```

## Acknowledgements
This project is based on the MMDetection framework. We thank the MMDetection team for their outstanding contributions.
