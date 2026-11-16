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
| Operating System | Linux (Rocky Linux 10.1) | This can be what ever you want, the environment has been produced using micromamba so as long as your system has your GPUs set up correctly (with cuda) then the environment libaries (e.g. pytorch) can make use of them everything should work.|

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

```bash
curl -Ls https://micro.mamba.pm/install.sh | bash
```
Reload your shell

```bash
source ~/.bashrc
```
Verify installation:
```bash
micromamba --version
```
### Create environment from environment.yml
Using the yml file included in this repository create a new micromamba environment

```bash
micromamba env create -f environment.yml
```
Once you have agreed to the installation of the required packages this will create a new micromamba environment called ```MTG_RoA_Satpy```. This environment contains most of what you need to run the scripts included here. However the Rain over Africa code will need to be installed to the environment seperately.

### Activate the environment
To install the RoA code you will first need to activate the environment.

```bash
micromamba activate MTG_RoA_Satpy
```

### Install RoA package from source
Now to install the RoA package to your environment, first change directory to the roa directory included in this repository. This directory is almost an exact copy of the roa repository. However, it has been recreated here to facilitate some minor changes and ensure that any modifications to the original code base don't produce any conflicts with with the environment.

```bash
cd roa
```

Once in the roa directory install the roa code to your active environment.

```bash
pip install -e .
```
Installing using the -e flag will allow you to make modifications to the roa package as required and these changes will be available to be used in your environment. The modifications to the made from the original roa code were mostly to allow for projection data to be written to the output files to allow for the output files which can be used directly with the ADAGUC server.

### Verify PyTorch GPU Access
Once RoA has been installed check that the PyTorch library has access to the GPU.

```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Expected result
```bash
True
```
## Input data
### MTG FCI
The MTG FCI files, with filenames such as this
```text
W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+FCI-1C-RRAD-FDHSI-FD--CHK-BODY---NC4E_C_EUMT_20261016000404_IDPFI_OPE_20261016000006_20261016000022_N__O_0001_0001.nc
```
will be available directly from the EUMETCAST Terrestrial service. These should be either stored in or symbolically linked to the ```/MTG_roa_LI/data/MTG_FCI_input``` directory. Once you have the appropriate input files present you should be able to run the ```ROA_MTG-specify-datetime.py``` script for the relevant date and time using the environment described above to start to generate Rain over Africa output.

The naming convention of masked files available through the EUMETCAST satellite transmission service is unknown to me at this time. However, the processing of files accessed in this way should still be possible. In this case to ensure that the processing is performed properly you should delete any existing regridding weights files in the regrid weights directory ```MTG_roa_LI/scripts/regrid_weights``` and run the ```ROA_MTG-specify-datetime.py``` script. This will take some extra time the first go as the pickle file containing the weights for conversion between MTG and MSG grids will have to be created.

### MTG LI
The MTG LI data is used to assist in the forward extrapolation of the retrieved rainfall product (RoA). It too should be available directly from the EUMETCAST Terrestrial service and will follow a very similar naming convection
```text
W_XX-EUMETSAT-Darmstadt,IMG+SAT,MTI1+LI-2-AF--FD--CHK-BODY--DIS-NC4E_C_EUMT_20261023134736_L2PF_OPE_20261015235000_20261015235030_N__O_0144_0001.nc
```
As with the FCI data the accumulated flashes MTG LI files should be copied or symbolically linked to their directory ```/MTG_roa_LI/data/MTG_LI_input``` so that the extrapolation script can see them. For an extrapolation to be able to occur there needs to be multiple consecutive RoA files available. This will enable the optical flow vectors to be generated. However, if there are MTG-LI files present for this time too then these will also be included in an attempt to improve the performance of the extrapolation.
As the extrapolation code relies on accumulating the lightning data to match the frequency of the rainfall data being gathered the only MTG LI data that is used is the accumulated flashes product (AF). All other MTG LI products will be ignored by the extrapolation code.

### Output directories
The ```ROA_MTG-specify-datetime.py``` script will automatically write RoA output files e.g. ```RoA_202610160030_WestAfrica.nc``` to an appropriate output directory ```MTG_roa_LI/data/roa_retrievals/$YYYYMMDD```. The extrapolation code will look in these directories to find the input files required to work out the optical flow vector field. The files produced have been modified from the standard RoA structure so that they are cf-compliant and meet the requirements to be processed by the ADAGUC server. Therefore, it makes sense that this output directory could be linked directly to the input directories for the ADAGUC server or automated jobs set up to copy files to the appropriate directory every time a new file is produced. As mentioned earlier, this output directory structure is required for the extrapolation step so automatically moving or deleting files from this directory could interfere with the production of extrapolation files. That said it is important to remember that cleanup scripts from this directory are required. If files are never removed from here it will cause the storage to run out and lead to system failure.

The ```extrapolation_RoA_LI.py``` script uses the files in the ```MTG_roa_LI/data/roa_retrievals/$YYYYMMDD``` directories to be able to perform the extrapolations. Similarly, once the extrapolations are complete the extrapolation output files e.g. ```RoA_202610160030_WestAfrica_010.nc, RoA_202610160030_WestAfrica_020.nc, RoA_202610160030_WestAfrica_030.nc, RoA_202610160030_WestAfrica_040.nc ...```   are written to ```MTG_roa_LI/data/roa_extrapolations/$YYYYMMDD``` directories. As with the RoA output files these have been tested to ensure that they are compatible with the ADAGUC server. Therefore they should be linked or copied to the appropriate input directory for the ADAGUC server.

### Domains 
If required, modify the domains.json file. Domains are set by providing a name of the domain, the centre lat and lon values and a vertical and horizontal size of the domain (this is based on Meteosat second generation pixels). If the domains file is modified the next time the code is run there will be  a slight delay due to the need to produce new weights files. However once this has been done once the process of regridding within the retrieval and extrapolation will be fast.

### First run
This will take longer than usual (as mentioned above) as the regridding information has to be produced. The regridding weights will be stored in the regrid_weights direcory and will consist of a pickle file which is for regridding the MTG data onto a MSG grid and several msg and mtg to and from cartesian weights. These second regridding weights files are required as pysteps (used for the forward extrapolation of data) works best on a cartesian grid. Both RoA (on the MSG grid) and MTG LI (on the MTG grid) need to regridded to an domain specific cartesian grid for extrapolation before results are regridded back before being written to file.

To run the RoA script ```ROA_MTG-specify-datetime.py``` make sure the micromamba environment is active and choose a date and time which you have data for, also make sure that you have updated the domains.json file appropriately for the domain or domains you wish to run.

Then run the script with the date and time as follows:

```bash
python ROA_MTG-specify-datetime.py $YYYYMMDDhhmm
```

Subsequent runs should take less time to process. On my system for testing I was able to process the West Africa and the Ghana domains in less than 40 seconds (both rainfall retrieval and extrapolations for 3 hours) including saving files.

### File monitoring for running code.
Not included here but for FASTA we make use of system.d and inotifywait scripts to detect when new satellite files are present and automatically trigger the running of RoA and extrapolation code.
