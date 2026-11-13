# Rain over Africa Extrapolation using Meteosat Third Generation
 
A Python workflow for generating **Rain over Africa (RoA)** precipitation estimates from **Meteosat Third Generation (MTG)** Flexible Combined Imager observations and producing short-term precipitation nowcasts using RoA data combined with MTG Lightning imager data making use of **pysteps** optical flow techniques.
 
## Overview
 
The original Rain over Africa (RoA) system was developed using observations from the Meteosat Second Generation (MSG) satellite series. This repository provides an implementation adapted for the **Meteosat Third Generation (MTG)** era, using data from the MTG Lightning Imager (LI) and MTG Flexible Combined Imager (FCI) products.
 
The workflow performs three key tasks:
 
1. Reads and processes MTG Flexible Combined Imager (FCI) observations.
2. Generates Rain over Africa (RoA) rainfall-rate fields compatible with the original methodology.
3. Uses both RoA rainfall fields and MTG LI observations with the **pysteps** nowcasting framework to produce short-range rainfall extrapolations.
 
The resulting products can be used for near-real-time monitoring and short-term forecasting of convective rainfall across Africa.
 
---
 
## Processing Workflow
 
The overall processing chain is illustrated below:
 
```text
                   MTG FCI Data
                        │
                        ▼
         Refactoring to MSG grid and channels
                        │
                        ▼
                RoA Rainfall Fields
                        │
                        ▼
                MTG LI data + RoA
                        │
                        ▼
                 Motion Estimation
                    (pysteps)
                        │
                        ▼
              Short-Term Extrapolation
                        │
                        ▼
                   NetCDF Output
                        │
                        ▼
        WMS availability through ADAGUC server
```

## Hardware Requirements
### System

The system that has been used for testing is significantly over-specced for this particular purpose. Due to this I have provided the hardware description of the system we are using but also some information about what is likely needed for operational use. In an attempt to estimate the requirements some tests have been performed and monitoring use of the system components to assess what the requirements truly are. Tests are based on a subdomain covering a West Africa.
  
| Component | Our specifications | Comments |
|------------|----------------|----------------|
| CPU | 24 CPUs Intel(R) Core(TM) Ultra 9 285K |This is likely far more CPUs than are needed, 8+ is likely suitable.|
| RAM | 124 GB and 31 GB Swap Memory | RoA is memory intensive. Max memory usage for a WA subdomain was 7.4 Gb therefore something like 12-16 Gb RAM is OK if larger domains are envisioned then more should be provided. |
| Storage | 3.6 TB SSD | All that is needed is sufficient space for input data and however much output data is required. SSD storage is preferred due to speed of reading and writing. |
| GPU | NVIDIA GeForce RTX 5090 GPU with CUDA support ( 32 GB VRAM) | System was also tested on NVIDIA L4 (24 GB VRAM) and NVIDIA RTX A4000 (16 GB VRAM) the VRAM usage for pan Africa domains is very large but for regional subdomains (e.g. WA) is the same as standard RAM usage (~7.4 GB ). Therefore generous VRAM is required. |
| Operating System | Linux (Rocky Linux 10.1) | This can be what ever you want, the environment has been produced using micromamba so as long as your system has your GPUs set up correctly (with cuda) then the environment libaries (e.g. pytorch) can make use of them.|

## Verify NVIDIA GPU and CUDA Installation
 
Before creating the processing environment, verify that the system can detect the installed NVIDIA GPU and CUDA runtime.

### Check GPU Detection

Run:
 
```bash
nvidia-smi
```
 
Expected output should display:
 
- NVIDIA driver version
- CUDA version
- One or more detected GPUs
 
For example:

```text
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.105.08             Driver Version: 580.105.08     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        Off |   00000000:02:00.0 Off |                  N/A |
|  0%   35C    P8              1W /  600W |      66MiB /  32607MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            1816      G   /usr/bin/gnome-shell                     46MiB |
+-----------------------------------------------------------------------------------------+
```
### Verify CUDA is Accessible
 
Run:
 
```bash
nvcc --version
```
Expected output should contain a CUDA release version, for example:
 
```text
Cuda compilation tools, release 13.0
```
 
If `nvcc` is not available but `nvidia-smi` reports a CUDA version, this usually indicates that the CUDA runtime is installed but the CUDA toolkit is not present. This is acceptable for most users unless they intend to compile CUDA software.
 
## Environment installation

If your system's GPU is detectable and CUDA is accessible then the next step is the installation of the environment. I have used micromamba to achieve this but if you prefer an alternative method then use what you prefer.

### Installation steps
1. Clone this github repo 
2. Install Micromamba
3. Create environment from environment.yml
4. Activate environment
5. Install RoA package from source
6. Run validation tests

### Clone the Repository
 
```bash
git clone https://github.com/earajr/MTG_roa_LI.git
cd MTG_roa_LI
```
### Install Micromamba
