import sys
import data
import os
import tensorflow as tf
import keras
from keras import layers, models
import numpy as np
from sklearn.metrics import classification_report

'''In order to run this script, you can open a terminal and use the command: $env:TF_ENABLE_ONEDNN_OPTS=0 in order to setup TensorFlow properly, 
then use the command: python src/shallowNN.py <dataset_name> <number_of_layers>
where <dataset_name> can be one of the following: iris, mnist, fashion_mnist, cifar10 and <number_of_layers> can be 1, 2, or 3. 
This will load the specified dataset, build and train a shallow neural network with the specified 
number of layers on it, and print out the accuracy and classification report.'''

def build_model(input_shape, num_classes, layers_count):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    
    # Add hidden layers based on layers_count
    for _ in range(layers_count):
        model.add(layers.Dense(64, activation='relu'))
    
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def main():
    if len(sys.argv) != 3:
        print("Usage: python src/shallowNN.py <dataset_name> <number_of_layers>")
        print("Available datasets: iris, mnist, fashion_mnist, cifar10")
        print("Number of layers: 1, 2, or 3")
        sys.exit(1)
    
    dataset_name = sys.argv[1].lower()
    layers_count = int(sys.argv[2])
    if layers_count not in [1, 2, 3]:
        print("Please specify 1, 2, or 3 layers.")
        sys.exit(1)

    # Load data
    if dataset_name == 'iris':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_iris()
    elif dataset_name == 'mnist':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_mnist()
    elif dataset_name == 'fashion_mnist':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_fashion_mnist()
    elif dataset_name == 'cifar10':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_cifar10()
    else:
        print(f"Dataset '{dataset_name}' not recognized.")
        sys.exit(1)

    # Ensure numeric dtypes and flatten images if needed (Keras Dense expects 2D inputs)
    if X_train.ndim > 2:
        X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.float32)
        X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.float32)
    else:
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int)

    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y_train))
    
    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    # Build model
    model = build_model(input_shape, num_classes, layers_count)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train
    print(f"Training a {layers_count}-layer shallow neural network on {dataset_name}...")
    model.fit(X_train, y_train_cat, epochs=10, batch_size=32, verbose=2)
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"Test accuracy: {accuracy*100:.2f}%")

    # Generate predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Print classification report for the labels present in the test set only
    labels = np.unique(y_test)
    target_names = [str(i) for i in labels]
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))

    # Clear Keras / TF state to avoid background threads preventing process exit
    try:
        keras.backend.clear_session()
    except Exception:
        pass

    # Return success exit code
    return 0

if __name__ == "__main__":
    sys.exit(main())