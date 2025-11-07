# Machine Learning Techniques Comparison on Multiple Datasets

This project compares various six machine learning techniques across four datasets in order to analyze how data characteristics influence model selection and performance. Across this repository, we aim to answer the following key questions:

- How do data trends influence model choice?
- Why might SVMs not work well on all classes of Iris?
- When is supervised learning preferable? When might unsupervised methods be used?

This GitHub repository was made in collaboration by [Yarah Al-Fouleh](https://github.com/yarah-yma1), [Senem Keceli](https://github.com/senem584), and [Anthony Nguyen](https://github.com/anturny).

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
- For detailed information about these datasets, please visit our [adjacent dataset repository](https://github.com/anturny/dataset_information) in tandem with this repository.

## Techniques
- Support Vector Machines (SVM)
    - Supervised learning models that find the optimal boundary to separate different groups of data (2D & 3D), maximizing the margin between datapoints of different classes. They are commonly used for classification and regression tasks, especially when the data has clear margins or when a decision boundary needs to be found in high dimensional or complex feature spaces.
    - Upon training in the [svmExp.py](/src/svmExp.py) file, we find the following:
        - ## Iris (Runtime: 2.374 seconds)
            - The Iris dataset consists of a small, well-balanced collection of numeric features with only three classes. Although it is highly separable in many cases, some classes, notably Versicolor and Virginica, are not perfectly linearly separable. This can pose a challenge for a simple linear SVM, which may struggle to distinguish these classes accurately without kernel tricks. While the dataset’s simplicity often allows SVMs to achieve perfect or near-perfect classification, the overlapping regions highlight the importance of kernel selection and parameter tuning to handle non-linear decision boundaries effectively.
            - ![alt_text](/media/svmClassReportIris.PNG)
        - ## MNIST (Runtime: 5 minutes)
            - MNIST is a large, high dimensional dataset of handwritten digits, which presents both opportunities and challenges for SVM classifiers. On the one hand, the dataset’s structure and clear digit shapes enable the SVM to perform very well, especially when using suitable kernels and hyperparameter tuning. However, the high dimensionality (each image has 784 features) can lead to computational complexity and longer training times. Furthermore, subtle variations in handwriting or noise can cause misclassification, so the model’s success heavily depends on choosing the right kernel and regularization parameters to balance accuracy and efficiency.
            - ![alt_text](/media/svmClassReportMnist.PNG)
        - ## Fashion MNIST (Runtime: 7 minutes)
            - Fashion MNIST introduces greater complexity due to the visual similarity of many clothing items, making class boundaries less distinct. Raw pixel data may not capture the nuanced features needed to differentiate between similar items like shirts and pullovers. As a result, a basic SVM with a linear kernel often struggles to achieve high accuracy, especially for classes that look alike or share similar textures. To improve performance, more sophisticated feature extraction or the use of kernel methods that can model non-linear relationships is necessary. Without these enhancements, the classifier may produce lower precision and recall for certain classes, reflecting the dataset’s inherent complexity.
            - ![alt_text](/media/svmClassReportFashionMnist.PNG)
        - ## CIFAR-10 (Runtime: 10 minutes)
            - CIFAR-10 is notably the most challenging among these datasets due to its high variability, small color images, and complex visual features. The significant intra-class variation (such as different angles and backgrounds) and interclass similarities make it difficult for an SVM trained on raw pixel data to achieve high accuracy. The model's reliance on pixel-level features makes it inadequate for capturing the hierarchical and spatial information essential to distinguish diverse objects like airplanes versus trucks. Consequently, the SVM performs close to random chance, indicating the need for more advanced models like convolutional neural networks, which can learn spatial hierarchies and extract meaningful features from the images.
            - ![alt_text](/media/svmClassReportCifar10.PNG)

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

- Each machine learning technique's .py file under the src folder will have their own set of instructions, but everything requires an initial run of the [data.py](src/data.py) file, which is seen as a class.

## References
This section will include any citations in IEEE format used to help create this repository.