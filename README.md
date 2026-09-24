# Evaluating the Impact of Data Augmentation on Medical Image Classification

Melanoma classification pipeline built to test whether classical image augmentation improves performance when the training set is already large. Four traditional classifiers are trained on the original and the augmented data and scored against the same held out test set.

---

## Dataset

[Melanoma Skin Cancer Dataset of 10,000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) (Kaggle)

| Property | Value |
|---|---|
| Classes | 2 (benign, malignant) |
| Resolution | 300 x 300 |
| Training images | 9,600 |
| Test images | 1,000 |
| After augmentation | ~19,000 training images |

The dataset is not included in this repository. Download it from Kaggle and place it as described below.

---

## Repository structure

```
.
├── Augmentation.py     # generates one augmented copy per training image
├── Merge.py            # merges original and augmented images into one training set
├── Evaluation.py       # PCA feature extraction, training, evaluation
├── train/
│   ├── benign/
│   └── malignant/
├── test/
│   ├── benign/
│   └── malignant/
└── README.md
```

`augmented_images/` and `augmented_train/` are created by the scripts.

---

## Method

### 1. Augmentation

`Augmentation.py` applies random geometric transformations using Keras `ImageDataGenerator`:

| Transformation | Setting |
|---|---|
| Rotation | up to 40 degrees |
| Width shift | 20% of image width |
| Height shift | 20% of image height |
| Shear | 20% intensity |
| Zoom | 20% factor |
| Horizontal flip | enabled |
| Fill mode | nearest neighbor |

One augmented copy is produced per original image, which doubles the training set.

### 2. Merging

`Merge.py` combines `train/` and `augmented_images/` into `augmented_train/`, preserving the class folder structure.

### 3. Feature extraction and classification

`Evaluation.py` resizes every image to 100 x 100, flattens it, and reduces dimensionality with PCA. Four classifiers are then trained and evaluated:

- Random Forest
- Support Vector Machine (linear kernel)
- Decision Tree
- Gaussian Naive Bayes

Set `TRAIN_FOLDER = 'train'` for the baseline run and `TRAIN_FOLDER = 'augmented_train'` for the augmented run. The test set, the split, and the random seed stay fixed across both, so the training data is the only variable.

---

## Results

Test accuracy on the same held out set of 1,000 images:

| Classifier | Original | Augmented | Change |
|---|---|---|---|
| Random Forest | **0.887** | 0.885 | −0.002 |
| SVM (linear) | 0.862 | **0.871** | +0.009 |
| Decision Tree | 0.841 | **0.847** | +0.006 |
| Naive Bayes | **0.826** | 0.793 | −0.033 |

Confusion matrices after augmentation:

| Classifier | TN | FP | FN | TP |
|---|---|---|---|---|
| SVM | 426 | 74 | 55 | 445 |
| Decision Tree | 417 | 83 | 70 | 430 |
| Random Forest | 436 | 64 | 51 | 449 |
| Naive Bayes | 475 | 25 | 182 | 318 |

Accuracy moved by less than one point in either direction, and the ranking of the classifiers was unchanged. The likely explanation is that the original dataset was already large and diverse enough that geometric augmentation added no new information.

---

## Generative augmentation

GAN and StyleGAN were implemented first with the intention of synthesizing new lesion images. Training was carried out past 1,000 epochs without reaching usable image quality or resolution within the available compute, and the approach was dropped in favor of classical transformations.

---

## Running it

```bash
pip install tensorflow scikit-learn scikit-image matplotlib seaborn numpy

python Augmentation.py    # creates augmented_images/
python Merge.py           # creates augmented_train/
python Evaluation.py      # set TRAIN_FOLDER first
```

---

## License

This repository is provided for academic and research purposes only.
