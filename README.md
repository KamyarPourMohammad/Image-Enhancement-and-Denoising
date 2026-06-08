# Image Processing Fundamentals: Enhancement, Equalization & Filtering

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.19+-orange.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.3+-red.svg)](https://matplotlib.org/)

## *Overview*

This repository contains comprehensive implementations of fundamental image processing techniques, including intensity transformations, histogram equalization, and spatial filtering for noise removal. The projects demonstrate both theoretical understanding and practical implementation of core computer vision concepts.


##  *Projects*

### 1. Intensity Transformations (`enhancement.py`)

Three fundamental spatial domain transformations for image enhancement:

| Transformation | Formula | Application |
|---------------|---------|-------------|
| **Log Transform** | s = log(1 + r) | Enhances dark regions, medical imaging |
| **Exponential Transform** | s = e^(βr) - 1 | Enhances bright regions |
| **Gamma Correction** | s = r^γ | Monitor calibration, exposure correction |

## 2. Histogram Equalization 
Implementation of histogram equalization using CDF (Cumulative Distribution Function) without built-in functions.

### Algorithm Steps:

1. Calculate histogram of input image

2. Compute probability distribution

3. Calculate Cumulative Distribution Function (CDF)

4. Create mapping: s = round(CDF × 255)

5. Apply mapping to each pixel

### Visualization Output:

1. Original image

2. Original histogram

3. Equalized image

4. Equalized histogram

## 3. Spatial Filtering for Noise Removal
Comparison of three spatial filters for Salt & Pepper Noise removal with 5×5 kernel size.

|Filter|	Type	|Best For	|Edge Preservation|
|---------------|---------|-------------|---|
|Median Filter|	Non-linear|	Salt & Pepper|	Excellent
|Mean Filter|	Linear|	Gaussian noise|	Poor|
|Gaussian Filter|	Linear|	Gaussian noise|	Moderate|

- Evaluation Metrics: PSNR (Peak Signal-to-Noise Ratio) - Quantitative measurement

## *Clone Repository*
```bash
git clone https://github.com/KamyarPourMohammad/Image-Enhancement-and-Denoising.git
cd Image-Enhancement-and-Denoising
