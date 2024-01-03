# RTQuicAnalysis
 
## Overview
The code featured in this project contains an AI approach to analyzing RT-Quic data. The dataset being analyzed can be found in the Data folder, featuring CSVs containing the dataset being used in a few different formats. The BigAnalysis folder contains the raw flouresence and feature-selected values in a uniform format. The other folders each pertain to a different data format with fewer wells in them.

## Code Overview (Current Version)
### **ML_Quic_Raw.py** 
This file contains all the logic for processing raw fluorescence data from the BigAnalysis files. The file operates as an interface between the different model types and the data, allowing for a simple object oriented approach to introducing new models or data. 

### Models
Each of the model files in the folder Models implements a different type of machine learning method to predict the data, providing a uniform interface to work with the ML-Quic operations without any alterations to the ML-Quic workflow. 

### Driver
The **Driver.py** and **Driver.ipynb** files both run the ML-Quic process for both the raw and analyzed data, providing a method of displaying the salient information for a given study without the clutter of a lot of code as the logic is all handled in the ML-Quic workflow. 
