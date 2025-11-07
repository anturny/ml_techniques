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

- Shallow Neural Networks (1-3 layers) (Runtime: 31.861 seconds)
    - Shallow neural networks are simple machine learning models with one or a few layers of neurons that learn to map input data to outputs by adjusting weights through training. They are typically used for basic pattern recognition tasks, small-scale problems, or when interpretability and quick training are more important than capturing complex data structures.
        - In this example, the MNIST dataset would be the most suitable choice to exemplify shallow neural networks because of its relatively simple, high contrast, greyscale images of handwritten digits, which can often be effectively classified with just a few layers of neurons. Its low complexity allows shallow networks (comprising one to three layers) to achieve reasonable accuracy without the need for deep architectures. However, using shallow neural networks on datasets like CIFAR-10 or Fashion-MNIST can present significant challenges; these datasets contain more complex, high-variability, and colorful images that require deeper networks to learn hierarchical features effectively. Shallow networks may struggle to capture the intricate patterns and spatial relationships in such data, leading to poor classification performance. Consequently, while shallow neural networks can serve as good educational tools for understanding basic concepts, they are generally insufficient for complex image data, where deeper architectures like convolutional neural networks (CNNs) are essential to achieve high accuracy and robust feature learning.
        - ![alt_text](/media/snnMNIST2layer.PNG)


- Deep Neural Networks (Runtime: 5 minutes)
    - Deep neural networks are complex machine learning models with multiple layers that automatically learn hierarchical features from data, enabling them to perform tasks like image recognition, natural language processing, and speech recognition. They are used when dealing with large, complex datasets that require capturing intricate patterns and relationships beyond the capabilities of shallow models.
        - The results clearly show that the accuracy on the training data (78%) is significantly higher than on the test data (42%), indicating the model has overfitted to the training set and does not generalize well to unseen data. As we increase the number of layers and epochs in a deep neural network, the model becomes more expressive and able to learn complex patterns, which often leads to higher training accuracy. However, this increased capacity can also cause the network to memorize the training examples rather than learn generalizable features, especially if training continues for many epochs. This phenomenon illustrates a common trade-off in deep learning: deeper networks with more epochs can achieve very high training accuracy but may not improve, or even worsen, test accuracy without proper regularization or data augmentation. It highlights the importance of balancing model complexity and training duration to avoid overfitting, ensuring the model captures generalizable features rather than noise in the training data.
        - ![alt_text](/media/dnnCIFAR1010layer100epoch.PNG)

- Convolutional Neural Networks (CNN)
    - Convolutional neural networks (CNNs) are a type of deep learning model designed to automatically extract and learn hierarchical features from grid-like data such as images, using convolutional layers that capture spatial patterns. They are primarily used in tasks like image and video recognition, object detection, and facial recognition, where understanding spatial relationships is crucial for accurate predictions.

- Decision Trees
    - Decision trees are a supervised learning method that model decisions by splitting data based on feature values, creating a tree-like structure for classification or regression. They are used in situations where interpretability is important, such as customer segmentation or medical diagnosis, and work well with structured data of varying complexity.

- Random Forest
    - Random forests are an ensemble learning method that build multiple decision trees using different subsets of data and features, then combine their predictions to improve accuracy and reduce overfitting. They are commonly used in classification and regression tasks where high performance and robustness are needed, such as in finance, bioinformatics, and feature rich datasets.

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

- Some files using TensorFlow may require the following text code in the Python environment in order to solve a float problem:
```
$env:TF_ENABLE_ONEDNN_OPTS=0
```

## References
This section will include any citations in IEEE format used to help create this repository.