# Machine Learning Techniques Comparison on Multiple Datasets

This project compares various six machine learning techniques across four datasets in order to analyze how data characteristics influence model selection and performance. Across this repository, we aim to answer the following key questions:

- How do data trends influence model choice?
- Why might SVMs not work well on all classes of Iris?
- When is supervised learning preferable? When might unsupervised methods be used?

## Table of Contents
- [Implementation](#implementation)
- [Datasets](#datasets)
- [Techniques](#techniques)
- [Requirements](#requirements)
- [How to Use](#how-to-use)
- [References](#references)

## Implementation
Machine learning is a subcategory of artificial intelligence that enables computers to learn from data and improve their performance on a task without being explicitly programmed for every specific case. In this repository, we compare machine learning techniques to analyze and classify different [datasets](#datasets), such as images and tabular data, by training models like those listed under [techniques](#techniques). The implementation involves loading and preprocessing the data, selecting appropriate model and parameters, train the models on the datasets, and evaluate their performance using various metrics. This will help us understand how different algorithms adapt to the data's characteristics, and how their performance varies on the complexity and nature of each dataset.

## Datasets
- **Iris:** Small tabular dataset with 3 classes of Iris species
- **CIFAR-10:** 60,000 color images in 10 classes at 32 x 32 resolution
- **MNIST:** 70,000 handwritten images of digits (0-9) at 28 x 28 resolution
- **Fashion MNIST:** 70,000 greyscale images of 10 clothing item classes at (28 x 28 resolution)

## Techniques
- Support Vector Machines (SVM)
    - Supervised models that find the optimal boundary to separate classes by maximixing the margin between data points.
- Shallow Neural Networks (1-3 layers)
    - Consists of 1-3 layers of interconnected neurons that learn simple patterns in data.
- Deep Neural Networks
    - More intricate than Shallow Neural Networks and have many layers that allow them to model complex and hierarchical features.
- Convolutional Neural Networks (CNN)
    - Specialized for image data and comparing convolutional layers to capture spatial features
- Decision Trees
    - Splits data based on feature thresholds to create a flowchart-like model for classification or regression.
- Random Forest
    - Ensemble of decision trees that improve accuracy and validation by combining multiple tree predictions.

## Requirements
This repository is mostly a read only repository. However, if you would like to follow along with the code, please have the following programs ready:

- Visual Studio Code (Software)
- Python Language on Computer (3.12.0)
- Gitbash (Optional)

- This project is designed to run in a VSCode terminal using a Python environment.

Use 'pip install -r requirements.txt' to install the required dependencies.

## How to Use
- To run this code, you will need to have a Python environment installed on your computer. It is recommended to use Visual Studio Code as this Python script was written and ran in VSCode. GitBash is also recommended in order to synchronize your VSCode with GitHub.
- In GitHub, click on the green icon labeled "<> CODE" on the top of this page and copy the HTTPS link.
- In VSCode, click on "Clone Git Repository" and paste the copied link from GitHub.
- In the search bar, type in "Python: Create Environment" and then select a preferred environment. This code used .venv as the virtual environment.
- When the virtual environment is open (appears as .venv in the list of items in the left menu), you may navigate to the various .py files containing machine learning technique codes and select it. At this point, you may open your terminal and install the pip requirements for the necessary libraries in order to execute the code. Then, you may hit "Run" on the top right hand corner to execute the code.

## References
This section will include any citations in IEEE format used to help create this repository.