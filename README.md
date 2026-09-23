# Melanoma Classification: Evaluating the Impact of Data Augmentation

A controlled comparison of four classical machine learning classifiers on melanoma skin cancer images, measuring whether geometric data augmentation improves classification performance.

---

## 📌 Overview

This project examines a common assumption in medical image classification: that augmenting the training set with geometrically transformed copies improves model performance. Four classifiers are trained twice under identical conditions, once on the original training set and once on a doubled set that includes augmented copies, isolating augmentation as the only variable.

---

## 📁 Repository Structure

```
├── code/
│     ├── Augmentation.py        # Generates augmented copies of training images
│     ├── Merge.py               # Merges original and augmented images
│     └── Evaluation.py          # PCA, classifier training, and evaluation
│
├── augmented_images/            # Augmented copies used in the experiment
│     ├── benign/
│     └── malignant/
│
└── README.md
```

The augmented images are included so the reported results can be reproduced exactly, since the geometric transformations are applied randomly.

---

## ▶️ Running the Code

The melanoma dataset is not included in this repository. Place the `train` and `test` folders in the repository root, each containing `benign` and `malignant` subfolders, then run:

```bash
python code/Augmentation.py    # creates augmented_images/
python code/Merge.py           # creates augmented_train/
python code/Evaluation.py      # trains and evaluates
```

For the baseline run, set `TRAIN_FOLDER = 'train'` in `Evaluation.py`.

---

## 🔬 Methodology

### **1. Data Augmentation**

Each training image produces one augmented copy using random geometric transformations:

| Transformation | Range |
|----------------|-------|
| Rotation | ±40° |
| Width / height shift | 20% |
| Shear | 20% |
| Zoom | 20% |
| Horizontal flip | enabled |

Empty regions introduced by transformation are filled using nearest-neighbour interpolation.

### **2. Dataset Merging**

Augmented copies are merged with the original training images, doubling the training set. The test set remains untouched.

### **3. Feature Extraction**

- Images resized to 100 × 100 and flattened
- **PCA** reduces dimensionality to 100 components
- PCA is fitted on the training set and applied to the test set

### **4. Classification**

Four classifiers are compared under identical conditions:

- Support Vector Machine (linear kernel)
- Decision Tree
- Random Forest
- Gaussian Naive Bayes

Evaluation covers accuracy, precision, recall, F1-score, and confusion matrices.

---

## 📈 Results

All classifiers were evaluated on a held-out test set of 1,000 images (500 benign, 500 malignant).

### Accuracy

| Classifier | Before | After | Δ |
|------------|-------:|------:|------:|
| Random Forest | **0.887** | 0.885 | −0.002 |
| SVM (linear) | 0.862 | **0.871** | +0.009 |
| Decision Tree | 0.841 | **0.847** | +0.006 |
| Naive Bayes | **0.826** | 0.793 | −0.033 |

### Weighted metrics after augmentation

| Classifier | Precision | Recall | F1 |
|------------|----------:|-------:|-----:|
| Random Forest | 0.89 | 0.89 | 0.88 |
| SVM (linear) | 0.87 | 0.87 | 0.87 |
| Decision Tree | 0.85 | 0.85 | 0.85 |
| Naive Bayes | 0.83 | 0.79 | 0.79 |

### Malignant class detection

Per-class recall on malignant lesions, the clinically consequential metric:

| Classifier | Before | After |
|------------|-------:|------:|
| Random Forest | 0.90 | 0.90 |
| SVM (linear) | 0.85 | **0.89** |
| Decision Tree | 0.84 | **0.86** |
| Naive Bayes | 0.78 | **0.64** |

---

### Analysis

**Augmentation produced no meaningful gain.** Three of four classifiers shifted by less than one percentage point, a margin consistent with run-to-run variance rather than genuine improvement.

**Random Forest led in both settings** and was unaffected by the additional training data, suggesting its ensemble of decision boundaries had already converged on the information available in the feature representation.

**Naive Bayes degraded substantially.** Its malignant recall fell from 0.78 to 0.64 while precision on that class rose to 0.93, meaning the classifier became conservative: it predicted malignant less often but was more confident when it did. In a diagnostic context this is the wrong direction, since missed malignant cases carry greater cost than false alarms. The independence assumption underlying Naive Bayes is sensitive to shifts in feature distribution, and the augmented copies altered those distributions.

**Why augmentation contributed little.** Features here are PCA components of flattened pixel intensities, which encode global brightness and colour patterns rather than spatial arrangement. Rotation, shifting, and flipping modify precisely the spatial properties this representation discards, so augmented images occupy nearly the same region of feature space as their originals. Geometric augmentation would be expected to contribute more under convolutional architectures, where learned filters are sensitive to the spatial structure these transformations affect.

---

## 🗂 Dataset

Melanoma skin cancer dataset, binary classification between benign and malignant lesions.
Test set: 1,000 images, evenly balanced across both classes.

---

## 📜 License

This repository is provided for academic and research purposes.
