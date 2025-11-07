import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import data

def main():
    if len(sys.argv) != 2:
        print("Usage: python src/svmExp.py *<dataset_name>*")
        print("Available datasets: iris, mnist, fashion_mnist, cifar10")
        sys.exit(1)
    
    dataset_name = sys.argv[1].lower()
    
    # Load data based on the dataset name
    if dataset_name == 'iris':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_iris()
    elif dataset_name == 'mnist':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_mnist()
    elif dataset_name == 'fashion_mnist':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_fashion_mnist()
    elif dataset_name == 'cifar10':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_cifar10()
    else:
        print(f"Dataset '{dataset_name}' is not recognized.")
        print("Available datasets: iris, mnist, fashion_mnist, cifar10")
        sys.exit(1)
    
    # Initialize and train the SVM classifier
    print(f"Training SVM on {dataset_name} data...")
    clf = SVC(gamma='scale')  # using default parameters; can be tuned
    clf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy on {dataset_name} test set: {accuracy * 100:.2f}%")

    # Print classification report for the labels present in this test set only
    labels = np.unique(y_test)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # Return normally so callers (including interactive shells) can decide how to exit
    return 0

if __name__ == "__main__":
    # Call main and exit the interpreter process with its return code
    sys.exit(main())