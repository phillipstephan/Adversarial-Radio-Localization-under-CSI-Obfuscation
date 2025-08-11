# CSI Obfuscation: Single-Antenna Transmitters Can Not Hide from Adversarial Multi-Antenna Radio Localization Systems

This repository contains the source code for the results presented in the paper

> <a href="https://arxiv.org/abs/2508.02553">Phillip Stephan, Florian Euchner, Stephan ten Brink: "CSI Obfuscation: Single-Antenna Transmitters Can Not Hide from Adversarial Multi-Antenna Radio Localization Systems"</a>

accepted for presentation at the 28th International Workshop on Smart Antennas 2025 in Erlangen, Germany.

## Prerequisites
Our code is based on Python, TensorFlow, NumPy, SciPy and Matplotlib.
Source files are provided as Jupyter Notebooks, which can be opened directly here on GitHub or using e.g. [JupyterLab](https://jupyter.org/).

We run our experiments on a JupyterHub server with NVMe storage, AMD EPYC 7262 8-Core Processor, 64GB RAM, and a NVIDIA GeForce RTX 4080 GPU for accelerating TensorFlow.
All indications of computation times are measured on this system.
It should also be possible to run our notebooks on less performant systems.

## How to Use the Code
The Jupyter Notebooks in this repository are numbered.
You must execute them in the right order.

* `0_DownloadDatasets.ipynb`: Downloads the required parts of the [dichasus-cf0x](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/) dataset for training and evaluation. **Note:** The dataset is large and not included in this repository; you must download it before running subsequent notebooks.
* `1_CSI_Obfuscation.ipynb`: Applies CSI obfuscation to the training and test sets, then applies the proposed recovery method. The processed CSI is saved as `.npy` files for later use.
* `2_AoA_Triangulation.ipynb`: Estimates angles of arrival from the CSI and performs classical triangulation, both with and without the recovery method. **Estimated runtime:** ~30 minutes.
* `3_Fingerprinting.ipynb`: Performs CSI fingerprinting on the processed datasets, comparing results with and without recovery. **Estimated runtime:** a few minutes with GPU acceleration.
* `4_ChannelCharting.ipynb`: Computes the fused dissimilarity matrix (combining angle-delay profile and timestamp information; see [tutorial](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/)), and trains a Siamese neural network for channel charting on the CSI data (with and without recovery). **Estimated runtime:** a few minutes with GPU acceleration.

## Citation

```
@misc{stephan2025csiobfuscationsingleantennatransmitters,
      title={{CSI Obfuscation: Single-Antenna Transmitters Can Not Hide from Adversarial Multi-Antenna Radio Localization Systems}}, 
      author={Phillip Stephan and Florian Euchner and Stephan ten Brink},
      year={2025},
      eprint={2508.02553},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2508.02553}, 
}
```

## Other Resources
* [Christoph Studer's Channel Charting Website](https://channelcharting.github.io/)
* [DICHASUS Website](https://dichasus.inue.uni-stuttgart.de/)
* [Our tutorial on dissimilarity metric-based Channel Charting](https://dichasus.inue.uni-stuttgart.de/tutorials/tutorial/dissimilarity-metric-channelcharting/)
* [Our paper on dissimilarity metric-based Channel Charting](https://arxiv.org/abs/2308.09539)
