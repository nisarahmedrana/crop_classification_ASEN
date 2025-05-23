# Attention-Based Ensemble Learning for Crop Classification (ASEN)

This repository provides the implementation of the **ASEN framework** proposed in the paper (under review in **Earth Systems and Environment**):

**"Attention-Based Ensemble Learning for Crop Classification Using Landsat 8-9 Fusion"**  
by Zeeshan Ramzan, Nisar Ahmed, Qurat-ul-Ain Akram, Shahzad Asif, Muhammad Shahbaz, Rabin Chakrabortty, Ahmed F. Elaksher

## 📌 Overview

The ASEN model leverages an ensemble of Multilayer Perceptrons (MLPs) and a novel attention mechanism to classify crops using fused Landsat 8-9 imagery and field-surveyed GPS data. The approach uses vegetation indices like NDVI, SAVI, NDRE, RECI and combines them with reflectance values for improved performance.

## 📁 Repository Structure

- `utils/`: Preprocessing, feature computation, and plotting functions.
- `models/`: Base MLP architecture and ASEN ensemble.
- `train/`: Scripts to train MLPs and the ASEN attention layer.
- `evaluation/`: Model evaluation and metric comparison.
- `notebooks/`: Example notebook walkthrough.
- `data/`: Dataset (user must request access to data from corresponding author).
