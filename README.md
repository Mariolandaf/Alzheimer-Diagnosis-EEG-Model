# Alzheimer's Diagnosis using EEG Data

This repository contains the implementation of a machine learning model designed to aid in the diagnosis of Alzheimer's disease using EEG (Electroencephalogram) data. The project employs both MATLAB and Python for EEG preprocessing and analysis, ultimately classifying the EEG signals using a Convolutional Neural Network (CNN).

## Project Overview
The model aims to assist healthcare professionals by analyzing EEG signals and determining whether the data corresponds to a patient with Alzheimer's, a healthy individual, or someone with other neurological conditions. The model offers a non-invasive way to potentially speed up the diagnosis process and improve early intervention.

## Key Features
- **EEG Preprocessing in MATLAB**:  
  EEG data is preprocessed using the EEGLAB library in MATLAB, which includes artifact removal, signal normalization, and event tagging.
  
- **Feature Extraction in Python**:  
  Various statistical features such as fractal dimensions, Hjorth parameters, and band powers are extracted from the preprocessed EEG data using Python.

- **Convolutional Neural Network (CNN)**:  
  A CNN model is used to classify the EEG data into three categories: Alzheimer's, Healthy, and Other Neurological Conditions.

- **Data Source**:  
  The EEG data used in this project is sourced from publicly available databases, focusing on patients diagnosed with Alzheimer's, healthy individuals, and other conditions.

## Technologies Used
- **MATLAB (EEGLAB)**:  
  For preprocessing EEG data, including noise reduction, signal filtering, and feature extraction.
  
- **Python (NumPy, SciPy, TensorFlow, Sklearn)**:  
  Used for advanced data analysis, feature extraction, and training the neural network.

![Alt Preprocessed EEG](Preprocessed_EGG.png)

*Figure 1. Recording of neural activity of an encephalography after preprocessing*

## Results
- The EEG preprocessing in MATLAB was successful. However, the classification model achieved only basic accuracy, which is insufficient for practical implementation. Despite this, the approach marks a step forward in exploring research options within this field. The use of neural networks is justified, with the expectation that expanding the dataset will lead to a more robust and reliable model in the future. This reinforces the potential of machine learning techniques in advancing Alzheimer's diagnosis.


## Future Work
- Expand the dataset to include more EEG samples, particularly for Alzheimer's patients.
- Improve model accuracy by tuning hyperparameters and trying alternative machine learning models.
- Develop an interactive interface for real-time diagnosis visualization.
