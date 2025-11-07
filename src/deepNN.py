import sys
import numpy as np
import keras
from keras import layers, models
from sklearn.metrics import classification_report
import data  # your data.py file

'''In order to run this script, you can open a terminal and use the command: $env:TF_ENABLE_ONEDNN_OPTS=0 in order to setup TensorFlow properly, 
then use the command: python src/deepNN.py <dataset_name> <number_of_layers> <epochs>
where <dataset_name> is one of: iris, mnist, fashion_mnist, cifar10
<num_of_layers> is an integer like 1, 2, or 3
<epochs> is the number of training epochs'''

def build_model(input_shape, num_classes, layers_count):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    # Add hidden layers
    for _ in range(layers_count):
        model.add(layers.Dense(128, activation='relu'))
    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def main():
    # Parse command-line arguments
    if len(sys.argv) != 4:
        print("Usage: python src/deepNN.py <dataset_name> <number_of_layers> <epochs>")
        print("Datasets: iris, mnist, fashion_mnist, cifar10")
        sys.exit(1)

    dataset_name = sys.argv[1].lower()
    layers_count = int(sys.argv[2])
    epochs = int(sys.argv[3])

    # Load dataset
    if dataset_name == 'iris':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_iris()
    elif dataset_name == 'mnist':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_mnist()
    elif dataset_name == 'fashion_mnist':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_fashion_mnist()
    elif dataset_name == 'cifar10':
        X_train, X_test, y_train, y_test = data.load_and_preprocess_cifar10()
    else:
        print("Unrecognized dataset")
        sys.exit(1)

    # Reshape if needed
    if X_train.ndim > 2:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    y_train = np.array(y_train, dtype=int)
    y_test = np.array(y_test, dtype=int)
    num_classes = len(np.unique(y_train))

    # Convert labels to categorical
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    input_shape = (X_train.shape[1],)

    # Build and compile model
    model = build_model(input_shape, num_classes, layers_count)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train
    print(f"\nTraining a {layers_count}-layer deep neural network on {dataset_name} for {epochs} epochs...")
    model.fit(X_train, y_train_cat, epochs=epochs, batch_size=32, verbose=2)

    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest accuracy: {accuracy*100:.2f}%")

    # Generate predictions for training data
    y_train_pred_probs = model.predict(X_train)
    y_train_pred = np.argmax(y_train_pred_probs, axis=1)

    # Generate predictions for test data
    y_test_pred_probs = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred_probs, axis=1)

    # Labels for classification report
    labels = np.unique(y_train)
    target_names = [str(i) for i in labels]

    # Classification report for training data
    print("\nClassification Report on Training Data:\n")
    print(classification_report(y_train, y_train_pred, labels=labels, target_names=target_names, zero_division=0))

    # Classification report for test data
    print("\nClassification Report on Test Data:\n")
    print(classification_report(y_test, y_test_pred, labels=labels, target_names=target_names, zero_division=0))

    # Clear session
    try:
        keras.backend.clear_session()
    except:
        pass

    return 0

if __name__ == "__main__":
    main()