# Bee_AR_Leemorf_2026

This repository contains the code used for the paper "Honey Bee Sound Fundamental Frequency Trajectory Estimation Using Adaptive AR Modelling".

## Repository Contents

### README.md

Project overview and short description of the code layout.

### MATLAB

The MATLAB folder contains the main signal-processing and verification scripts:

- [ktoa.m](MATLAB/ktoa.m) - converts Lee-Morf coefficients to AR polynomial coefficients.
- [lee_morf_new.m](MATLAB/lee_morf_new.m) - implementation of the Lee-Morf adaptive AR tracking algorithm.
- [LEEMORF_SENSITIVITY_SYNTHETIC.m](MATLAB/LEEMORF_SENSITIVITY_SYNTHETIC.m) - sensitivity analysis over AR order and effective time window on a synthetic signal.
- [LEEMORF_VERIFICATION_DATASET.m](MATLAB/LEEMORF_VERIFICATION_DATASET.m) - dataset verification script comparing Lee-Morf with other pitch trackers.
- [LEEMORF_VERIFICATION_DATASET_WHITENESS.m](MATLAB/LEEMORF_VERIFICATION_DATASET_WHITENESS.m) - residual whiteness check for the dataset-based Lee-Morf output.
- [LEEMORF_VERIFICATION_SYNTHETIC.m](MATLAB/LEEMORF_VERIFICATION_SYNTHETIC.m) - synthetic-signal verification across multiple SNR levels.
- [Stability_AR_ORDER_CHECK.m](MATLAB/Stability_AR_ORDER_CHECK.m) - AR order stability study using information-theoretic selection.
- [Stability_length_study.m](MATLAB/Stability_length_study.m) - quasi-stationarity study based on Renyi entropy for different window lengths.
- [Stability_length_study_result_process.m](MATLAB/Stability_length_study_result_process.m) - post-processing and visualization of the length-stability results.

These scripts cover the quasi-stationarity study based on Renyi entropy, the Lee-Morf algorithm implementation, dataset extraction and verification, synthetic-signal validation, sensitivity analysis, and whiteness testing.

### Python

The Python folder contains the machine-learning experiment driver:

- [MAIN_ANALYSIS.py](Python/MAIN_ANALYSIS.py) - main executive script for the classification experiment reported in the paper.
