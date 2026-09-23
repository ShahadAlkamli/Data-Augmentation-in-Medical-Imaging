"""
Melanoma classification using PCA feature extraction and four classical
classifiers. Compares performance on the original versus the augmented
training set.

Set TRAIN_FOLDER to 'train' for the baseline run, or
'augmented_train' for the augmented run.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io, transform
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths
TRAIN_FOLDER = 'augmented_train'
TEST_FOLDER = 'test'

IMAGE_SIZE = (100, 100)
N_COMPONENTS = 100
RANDOM_STATE = 42


def load_and_resize_images(folder, target_size=IMAGE_SIZE):
    """Load every image under folder, resize, flatten, and label it."""
    images, labels = [], []

    for subfolder in sorted(os.listdir(folder)):
        subfolder_path = os.path.join(folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            if filename.startswith('.'):
                continue
            img = io.imread(os.path.join(subfolder_path, filename))
            img_resized = transform.resize(img, target_size)
            images.append(img_resized.flatten())
            labels.append(1 if subfolder.lower() == 'malignant' else 0)

    return np.array(images), np.array(labels)


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load data
    print(f"Loading training images from {TRAIN_FOLDER}...")
    X_train, y_train = load_and_resize_images(TRAIN_FOLDER)

    print(f"Loading test images from {TEST_FOLDER}...")
    X_test, y_test = load_and_resize_images(TEST_FOLDER)

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set:     {X_test.shape[0]} images")

    # Dimensionality reduction
    print(f"\nApplying PCA ({N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained:.3f}")

    # Classifiers
    classifiers = {
        'SVM': svm.SVC(kernel='linear', random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_pca, y_train)

        predictions = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = (accuracy, predictions)

        print(f"{name} test accuracy: {accuracy:.3f}")
        print(classification_report(y_test, predictions, zero_division=0))

    # Confusion matrices
    for name, (_, predictions) in results.items():
        plot_confusion_matrix(y_test, predictions, f'{name} Confusion Matrix')

    # Accuracy comparison
    names = list(results.keys())
    accuracies = [results[n][0] for n in names]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=names, y=accuracies, hue=names, palette='viridis', legend=False)
    plt.title('Test Set Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()

    print('\nSummary:')
    for name, (accuracy, _) in results.items():
        print(f"  {name:<15} {accuracy:.3f}")        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            if filename.startswith('.'):
                continue
            img = io.imread(os.path.join(subfolder_path, filename))
            img_resized = transform.resize(img, target_size)
            images.append(img_resized.flatten())
            labels.append(1 if subfolder.lower() == 'malignant' else 0)

    return np.array(images), np.array(labels)


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Load data
    print(f"Loading training images from {TRAIN_FOLDER}...")
    X_train, y_train = load_and_resize_images(TRAIN_FOLDER)

    print(f"Loading test images from {TEST_FOLDER}...")
    X_test, y_test = load_and_resize_images(TEST_FOLDER)

    print(f"Training set: {X_train.shape[0]} images")
    print(f"Test set:     {X_test.shape[0]} images")

    # Dimensionality reduction
    print(f"\nApplying PCA ({N_COMPONENTS} components)...")
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum()
    print(f"Explained variance: {explained:.3f}")

    # Classifiers
    classifiers = {
        'SVM': svm.SVC(kernel='linear', random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'Naive Bayes': GaussianNB(),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train_pca, y_train)

        predictions = clf.predict(X_test_pca)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = (accuracy, predictions)

        print(f"{name} test accuracy: {accuracy:.3f}")
        print(classification_report(y_test, predictions, zero_division=0))

    # Confusion matrices
    for name, (_, predictions) in results.items():
        plot_confusion_matrix(y_test, predictions, f'{name} Confusion Matrix')

    # Accuracy comparison
    names = list(results.keys())
    accuracies = [results[n][0] for n in names]

    plt.figure(figsize=(8, 6))
    sns.barplot(x=names, y=accuracies, hue=names, palette='viridis', legend=False)
    plt.title('Test Set Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.show()

    print('\nSummary:')
    for name, (accuracy, _) in results.items():
        print(f"  {name:<15} {accuracy:.3f}")
