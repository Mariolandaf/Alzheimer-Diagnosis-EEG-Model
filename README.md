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

## How to Run the Project
### Requirements
- MATLAB (with EEGLAB)
- Python 3.x
- Required Python libraries: `numpy`, `scipy`, `tensorflow`, `sklearn`

### Steps
1. Preprocess the EEG data in MATLAB using the provided scripts.
2. Save the preprocessed data in `.mat` format.
3. Run the Python script to extract features and train the CNN model.
4. Evaluate the model's performance using the classification report and confusion matrix.

## Results
- The model achieved a basic classification accuracy, but the results highlight the need for more extensive EEG datasets to improve performance. Future iterations will focus on obtaining more data and refining the model.

## Future Work
- Expand the dataset to include more EEG samples, particularly for Alzheimer's patients.
- Improve model accuracy by tuning hyperparameters and trying alternative machine learning models.
- Develop an interactive interface for real-time diagnosis visualization.

## Contributions
Contributions to improve the model's accuracy or suggestions for alternative methods are welcome. Please submit a pull request or open an issue for discussion.

## License
This project is licensed under the MIT License.
