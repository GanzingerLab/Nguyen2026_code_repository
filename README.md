# Nguyen2026_code_repository

This repository contains the image processing and quantitative analysis code developed for Nguyen et al. (2026).

## Authors

- Gerard Castro Linares
- Christian Niederauer

## 1. Repository overview

The repository contains four main components:

| Directory | Description |
| --- | --- |
| `Calcium_flux` | Quantification and classification of intracellular calcium dynamics. |
| `NFAT_analysis` | Quantification of NFAT nuclear translocation. |
| `CD19-ZAP70_analysis` | Single-particle tracking, colocalisation, cluster analysis and quantification of immune synapse dynamics. |
| `Maturation_training` | Training dataset, training code and pretrained convolutional neural network for immune synapse maturation classification. |

## 2. System requirements

### Operating system and tested versions

The analysis code was developed and tested on Microsoft Windows
(OS build 26300) using Python 3.10.16.

The scripts are organised into interactive code cells (`#%%`) and can
be executed using Spyder or Visual Studio Code.

### Hardware

The analysis was performed on a desktop computer with the following
specifications:

- Processor: Intel CPU
- Physical CPU cores: 16
- Logical CPU cores: 22
- RAM: 32 GB

These specifications describe the system used for testing and do not represent minimum hardware requirements.

Memory requirements and execution times depend on the size of the microscopy recordings and the analysis parameters.

No specialised hardware is required to run the analysis workflows.

### Software dependencies

The main Python dependencies and their tested versions are:

| Package | Version |
| --- | --- |
| Python | 3.10.16 |
| NumPy | 1.24.3 |
| pandas | 2.1.1 |
| SciPy | 1.11.2 |
| scikit-image | 0.25.0 |
| scikit-learn | 1.6.1 |
| Matplotlib | 3.7.3 |
| Seaborn | 0.13.2 |
| Trackpy | 0.6.4 |
| Numba | 0.58.0 |
| nd2 | 0.7.2 |
| nd2reader | 3.3.0 |
| tifffile | 2024.12.12 |
| ImageIO | 2.37.0 |
| Pillow | 9.5.0 |
| OpenCV | 4.11.0.86 |
| PyTables | 3.10.1 |
| h5py | 3.12.1 |
| TensorFlow | 2.11.1 |
| Keras | 2.11.0 |
| read-roi | 1.6.0 |
| PyYAML | 6.0.1 |
| tqdm | 4.66.1 |
| statannotations | 0.7.2 |
| statsmodels | 0.14.4 |

The CD19 and ZAP70 analysis additionally relies on the following
packages developed by the Ganzinger Lab:

- [SPIT](https://github.com/GanzingerLab/SPIT)
- [Inmune_cell_TIRF_analysis](https://github.com/GanzingerLab/Inmune_cell_TIRF_analysis)
  - Manuscript-specific branch: Nguyen_et_al

ImageJ was used to manually delineate cell regions of interest (ROIs).

## 3. Installation

The analysis workflows are designed to run within a shared Conda
environment based on the SPIT installation.

### 3.1. Prerequisites

Install the following software:

- [Anaconda](https://www.anaconda.com/)
- [Git](https://git-scm.com/)
- [GitHub Desktop](https://desktop.github.com/)
- [Picasso Desktop](https://github.com/jungmannlab/picasso)
- [ImageJ](https://imagej.net/ij/)

GitHub Desktop is optional if the repositories are cloned directly
using Git.

### 3.2. Installation of SPIT

Follow the installation instructions provided in the
[SPIT repository](https://github.com/GanzingerLab/SPIT).

The installation requires Picasso, naclib and SPIT.

Clone the following repositories:

```bash
git clone https://github.com/jungmannlab/picasso.git
git clone https://github.com/edovanveen/naclib.git
git clone https://github.com/GanzingerLab/SPIT.git
```

Navigate to the Picasso directory and check out the version
compatible with SPIT:

```bash
cd picasso
git checkout 491ae28
cd ..
```

Navigate to the SPIT directory and create the Conda environment:

```bash
cd SPIT
conda env create -f environment.yml
conda activate SPIT
cd ..
```

Install Picasso:

```bash
cd picasso
pip install .
cd ..
```

Install naclib:

```bash
cd naclib
pip install .
cd ..
```

Install SPIT:

```bash
cd SPIT
pip install -e .
cd ..
```

Install largestinteriorrectangle:

```bash
pip install largestinteriorrectangle
```

Optionally, install Spyder to execute the analysis scripts
interactively:

```bash
conda install spyder
```

### 3.3. Installation of postSPIT

The CD19 and ZAP70 analysis requires the [Inmune_cell_TIRF_analysis](https://github.com/GanzingerLab/Inmune_cell_TIRF_analysis) package.

A dedicated branch of this repository contains the implementation used for the analysis presented in Nguyen et al. (2026).

Make sure the SPIT Conda environment is activated:

```bash
conda activate SPIT
```

Clone the repository:

```bash
git clone https://github.com/GanzingerLab/Inmune_cell_TIRF_analysis.git
cd Inmune_cell_TIRF_analysis
```

Switch to the manuscript-specific branch:

```bash
git checkout Nguyen_et_al
```

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
```

Note that postSPIT also imports SPIT and Picasso functionality. Both packages must be installed in the same Python environment, using the Picasso version required by SPIT, as described in Section 3.2.

### 3.4. Additional Python dependencies

Most of the Python dependencies required for the analysis are
installed automatically during the installation of SPIT and postSPIT.

Additional packages are required for reading Nikon ND2 microscopy
recordings, image processing and immune synapse maturation
classification.

With the SPIT Conda environment activated, install the following
packages:

```bash
python -m pip install nd2==0.7.2 Pillow==9.5.0 tensorflow==2.11.1
```

The versions specified correspond to those used in the analysis
presented in Nguyen et al. (2026).

### 3.5. Typical installation time

The estimated installation time is approximately 30–60 minutes on a standard desktop computer, assuming that the prerequisite software is already installed.

Installation time may vary depending on the internet connection, hardware specifications and Conda dependency resolution.

## 4. Data availability

The complete experimental imaging dataset associated with Nguyen et al. (2026) is deposited in the BioImage Archive.

- Accession number: S-BIAD4491
- Repository link: [BioImage Archive – S-BIAD4491](https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4491)

The deposited dataset includes:

- Calcium flux microscopy recordings (.nd2).
- NFAT nuclear translocation microscopy recordings (.nd2).
- CD19 and ZAP70 dual-colour TIRF microscopy recordings (.raw).
- Associated microscope acquisition metadata for the TIRF recordings.
- Manually delineated ImageJ cell ROIs (.roi) for the CD19 and ZAP70
  analysis.

The original directory structure should be preserved when downloading the data, as experimental metadata are extracted from directory and file names.

Intermediate processed images and quantitative analysis results can be regenerated using the provided analysis code.

The training dataset and pretrained immune synapse maturation classification model are included in the `Maturation_training` directory of this repository.

Users can analyse individual recordings from the deposited dataset
without downloading the complete dataset.

## 5. General instructions for use

The analysis scripts are divided into sequential sections using the `#%%` convention, allowing individual processing and analysis steps to be executed independently in compatible Python development environments.

Before running each analysis, users should modify the input and output paths to point to their local data directories and check the corresponding analysis parameters.

The following sections describe the individual analysis workflows, their required inputs and expected outputs.

## 6. Calcium flux analysis

### Overview

This pipeline quantifies intracellular calcium dynamics from time-lapse fluorescence microscopy recordings of Fluo-4 AM-loaded cells.

### Input data

Original microscopy recordings in Nikon ND2 format, containing fluorescence and brightfield images.

### Analysis workflow

1. Run `Calcium_flux/cell_detection_CaFlux.py` to detect and track individual cells and extract their fluorescence intensities over time.
2. Run `Calcium_flux/Analysis_script.py` to process the fluorescence intensity traces, detect calcium peaks, classify calcium responses and generate quantitative results and plots.

The `utils.py` file contains supporting functions for signal processing, classification, statistical analysis and plotting. It does not need to be executed independently.

### Expected output

- Cell detection and tracking results.
- Individual cellular fluorescence intensity traces.
- Calcium peak detection and characterisation.
- Classification of calcium responses.
- Summary statistics and plots.

## 7. NFAT nuclear translocation analysis

### Overview

This pipeline quantifies NFAT nuclear translocation by measuring NFAT-eGFP fluorescence in the nuclear and cytoplasmic compartments of individual cells over time.

### Input data

Original microscopy recordings in Nikon ND2 format containing NFAT-eGFP and Hoechst fluorescence channels.

### Analysis workflow

Run `NFAT_analysis/NFAT_analysis.py`.

The script performs:

1. Nuclear and cellular segmentation.
2. Cell tracking.
3. Quantification of nuclear and cytoplasmic NFAT fluorescence.
4. Quantification of NFAT nuclear translocation.
5. Calculation of the proportion of responding cells.
6. Generation of summary statistics and plots.

### Expected output

- Cell segmentation and tracking results.
- Nuclear and cytoplasmic fluorescence measurements.
- NFAT nuclear translocation traces.
- Classification of cellular responses.
- Summary statistics and plots.


## 8. CD19 and ZAP70 dynamics analysis

### Overview

This pipeline analyses dual-colour TIRF microscopy recordings to characterise CD19 and ZAP70 dynamics, molecular colocalisation, cluster organisation and immune synapse maturation.

### Input data

- Original microscopy recordings (`.raw`).
- Associated microscope acquisition metadata.
- Spatial registration calibration files (find in the SPIT repository).
- Manually delineated ImageJ cell ROIs (`.roi`).
- Pretrained immune synapse maturation classification model.

### Analysis workflow

The analysis scripts are located in `CD19-ZAP70_analysis`.

#### Step 1 — Image preprocessing and single-particle tracking

Run `1.Single_particle_tracking.py`.

This script performs fluorescence channel separation, affine registration, cropping, single-particle localisation and tracking.

Cell ROIs are used to restrict the analysis to individual cells.

For the deposited dataset, manually delineated ROIs are already provided within the corresponding `output` directory. These ROIs were generated on the affine-corrected images and should be used with the corresponding registration and cropping parameters.

#### Step 2 — Cell-level analysis

Run `2.Cell_analysis.py`.

This script performs cell-specific analysis using postSPIT, including immune synapse maturation classification using the pretrained convolutional neural network.

#### Step 3 — Molecular dynamics analysis

Run `3. speeds_intensity_and_direction_analysis.py`.

This script quantifies molecular velocities, fluorescence intensities and directional motion of CD19 and ZAP70 trajectories.

#### Step 4 — Synapse and cluster-level analysis

Run `4. Whole_synapse_integrated_intensity_and_cluster_level_analysis_Maturing_VS_non-maturing.py`.

This script quantifies fluorescence intensities and cluster-level properties and compares maturing and non-maturing immune synapses.

#### Step 5 — Quantitative analysis and plotting

The following scripts perform additional trajectory analysis, statistical comparisons and plotting:

- `5. average_tracks.py`
- `5.TIRF_plotting_density.py`
- `average_tracks.py`

### Expected output

- Spatially registered, channel-separated TIFF image stacks.
- Single-particle localisation and tracking tables.
- Cell-specific trajectories.
- CD19–ZAP70 colocalisation results.
- Immune synapse maturation classifications.
- Cluster-level fluorescence measurements.
- Molecular velocity, intensity and directionality measurements.
- Summary statistics and plots.

### Analysis of new data

For new recordings, cell ROIs can be manually generated using the freehand selection tool in ImageJ after affine correction and channel separation. Save the ROIs in the corresponding output directories before continuing with cell-specific analysis.

## 9. Immune synapse maturation classifier

### Overview

The `Maturation_training` directory contains the training dataset and code used to develop the convolutional neural network for immune synapse maturation classification.

The classifier is based on the MobileNetV2 architecture.

### Training

The script `training_maturatration_classifier.py` performs data preparation, model training and validation.

To retrain the model, modify the training dataset directory in the script and execute the training procedure.

### Pretrained model

The pretrained model is provided with the repository and can be used directly for immune synapse maturation classification.

Retraining the model is not required to analyse the demo datasets or reproduce the classification procedure used in the manuscript.

The `recreate_maturation_plot.py` script provides additional visualisation of maturation classification results.

### Expected output

- Trained classification model.
- Training and validation metrics.
- Immune synapse maturation classification results.


## Expected runtime

The execution time depends on the number and dimensions of the microscopy recordings, the number of detected cells and particles, the analysis parameters and the available computational resources.

The calcium flux and NFAT analysis workflows are computationally less demanding than the CD19 and ZAP70 analysis, which involves image registration, single-particle localisation, tracking,colocalisation and cluster analysis.

Processing individual recordings is considerably faster than processing the complete experimental dataset.

The pretrained maturation classification model is provided, and model retraining is not required for the analysis.

Execution times have not been benchmarked.

## 10. License

This repository is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) License](LICENSE).

## 11. Citation

If you use this code or the accompanying datasets, please cite:

The manuscript is currently under submission. The final publication
reference will be added once the article is published.

In the meantime, please cite the bioRxiv preprint:

[Nguyen et al. (2026)](https://doi.org/10.64898/2026.10.09.757739)