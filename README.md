# AI-QuIC/QuICSeedAI
 
## Overview
This reposititory contains a set of classes and methods designed to aid in the analysis of Seed Amplification Assays (SAAs). The approach outlined here utilizes various machine learning methods and allows for comparison between different models (both supervised and unsupervised). The predictions of these models in their current form can be useful for identifying whether each reaction in a given well is positive, negative, or false positive. More information on the models used in this repostitory and a detailed description of the analysis can found in the corresponding paper, [AI-QuIC](https://doi.org/10.1101/2024.10.16.618742).

## Dataset
All of the data used in the analysis found here and in the AI-QuIC paper was generated by Real-Time Quaking-Induced Conversion (RT-QuIC) applied to detect seeding activity produced by Chronic Wasting Disease (CWD). The dataset was sourced from [this paper](https://www.biorxiv.org/content/10.1101/2024.07.23.604851v1) on comparing different decontamination techniques for surfaces used in venison processing.

## Code Overview (Current Version)
### **QuICSeedIF.py** 
This file defines a class called QuICSeedIF which acts as an intermediate step for uniform, easily reproducible inputs and outputs to models made with different packages and structures. This uniformity allows for easy implementation of new model types from packages such as SciKit-Learn, Keras, Pytorch, and more. The code for different training/testing splits, import of datasets from multiple files, evaluation of models, and plotting of outcomes is all contained within this class. Utilizing this package involves creating an associated object and using the desired methods to obtain outputs or run different functions.

### Models
Each of the model files in the folder Models implements a different type of machine learning method to predict the data, providing a uniform interface to work with the QuICSeedIF operations without any alterations to the QuICSeedIF workflow. This allows new models to be added for evaluation with only minimal extra work required to ensure it matches the QuICSeedIF standard model implementation (given in the template).

### Demo
The **Driver.py** and **Demo.ipynb** files both run the QuICSeedIF process for both the raw and analyzed data, providing a method of displaying the salient information for a given study without the clutter of a lot of code as the logic is all handled in the QuICSeedIF workflow. These files contain all the necessary steps to obtain meaningful training and/or analysis of AI models using the process provided by QuICSeedIF. Specifically the **Demo.ipynb** notebook outlines the process of using **QuICSeedIF** meaningfully to reproduce the results found in the related manuscript.
