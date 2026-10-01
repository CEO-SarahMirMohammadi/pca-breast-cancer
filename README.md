# PCA Breast Cancer

A machine learning project demonstrating **Principal Component Analysis (PCA)** on the **Wisconsin Breast Cancer dataset** using Scikit-learn.

The project modifies Example 3.10, which originally applies PCA to the Iris dataset, and adapts the same workflow to breast cancer data.

The implementation includes:

* Loading the Wisconsin Breast Cancer dataset
* Exploring the original feature space
* Visualizing the original data
* Standardizing the features
* Applying Principal Component Analysis
* Reducing the data to three principal components
* Measuring explained variance
* Visualizing the first two principal components

## Overview

The original Example 3.10 demonstrates PCA using the Iris dataset.

This project replaces the Iris dataset with Scikit-learn's **Wisconsin Breast Cancer dataset**.

The workflow is:

```text
Load Breast Cancer Dataset
          ↓
Extract Features and Labels
          ↓
Visualize Original Data
          ↓
Standardize Features
          ↓
Apply PCA
          ↓
Transform Data
          ↓
Analyze Explained Variance
          ↓
Visualize PCA Components
```

The main purpose of the project is to demonstrate how PCA can transform a dataset with many original features into a smaller number of principal components while retaining as much information as possible.

---

## Principal Component Analysis

### What is PCA?

**Principal Component Analysis (PCA)** is an unsupervised dimensionality-reduction technique.

PCA transforms the original features into a new set of variables called **principal components**.

The first principal component captures the greatest amount of variance in the data.

The second principal component captures the greatest remaining variance while being orthogonal to the first component.

Additional components continue this process.

Conceptually:

```text
Original Features
       │
       │
       ▼
   Standardization
       │
       ▼
       PCA
       │
       ▼
Principal Components
       │
       ├── PCA1
       ├── PCA2
       └── PCA3
```

---

## Dataset

This project uses the **Wisconsin Breast Cancer dataset** provided by Scikit-learn through:

```python
datasets.load_breast_cancer()
```

The dataset contains:

* **569 samples**
* **30 numerical features**
* **2 target classes**

The two target classes represent:

```text
0 → Malignant
1 → Benign
```

The features are numerical measurements computed from digitized images of breast mass cell nuclei.

Examples include:

* Mean radius
* Mean texture
* Mean perimeter
* Mean area
* Mean smoothness
* Mean compactness
* Mean concavity
* Mean symmetry
* Mean fractal dimension

There are also corresponding standard error and "worst" measurements.

---

## Why PCA?

The dataset contains **30 features**.

Visualizing all 30 dimensions simultaneously is difficult.

PCA provides a way to transform the original 30-dimensional feature space into a smaller space.

In this project, the transformation produces:

```text
30 original features
        ↓
      PCA
        ↓
3 principal components
```

This makes the data easier to visualize and analyze.

---

## Feature Standardization

Before applying PCA, the features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Standardization transforms the features so that they have approximately:

```text
Mean = 0
Standard deviation = 1
```

This is important because the original breast cancer features have different numerical scales.

Without standardization, features with larger numerical ranges can have a disproportionate influence on the PCA result.

The workflow therefore becomes:

```text
Original Data
     ↓
StandardScaler
     ↓
Standardized Data
     ↓
PCA
```

---

## PCA Implementation

The PCA model is created with three components:

```python
pca = decomposition.PCA(n_components=3)
```

The model is fitted to the standardized data:

```python
pca.fit(X_scaled)
```

The original data is then transformed into the new PCA space:

```python
X1 = pca.transform(X_scaled)
```

The resulting matrix has three columns:

```text
PCA1
PCA2
PCA3
```

---

## Explained Variance

One of the most important outputs of PCA is the **explained variance ratio**.

The program displays:

```python
print(pca.explained_variance_ratio_)
```

This shows the proportion of the total variance captured by each principal component.

For example, conceptually:

```text
PCA1 → largest share of variance
PCA2 → second-largest share
PCA3 → third-largest share
```

The project also calculates the total variance explained by the three selected components:

```python
np.sum(pca.explained_variance_ratio_)
```

This helps determine how much information from the original 30-dimensional dataset is represented by the three-dimensional PCA representation.

---

## Visualization

The project produces two visualizations.

### 1. Original Data

The first visualization plots the first two original features:

```python
plt.scatter(X[:, 0], X[:, 1], c=y)
```

The points are colored according to their target class.

The axes correspond to:

```text
X-axis → mean radius
Y-axis → mean texture
```

This provides a simple view of the original feature space.

### 2. PCA Data

The second visualization plots:

```python
plt.scatter(X1[:, 0], X1[:, 1], c=y)
```

The axes are:

```text
X-axis → PCA1
Y-axis → PCA2
```

This allows the transformed data to be viewed using the two most important principal components among the three calculated components.

---

## Project Structure

```text
pca-breast-cancer/
│
├── pca_breast_cancer.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Requirements

| Requirement  | Purpose                           |
| ------------ | --------------------------------- |
| Python 3.x   | Programming language              |
| NumPy        | Numerical operations              |
| Matplotlib   | Data visualization                |
| Scikit-learn | Dataset, standardization, and PCA |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/pca-breast-cancer.git
```

Move into the project directory:

```bash
cd pca-breast-cancer
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## Running the Program

Run the Python program:

```powershell
python pca_breast_cancer.py
```

The program will:

1. Load the breast cancer dataset.
2. Display the dataset dimensions.
3. Display the available feature names.
4. Plot the original feature space.
5. Standardize the features.
6. Create a PCA model with three components.
7. Transform the standardized data.
8. Display the transformed dataset shape.
9. Display the explained variance ratio.
10. Calculate the total variance explained by the three components.
11. Plot the PCA representation.

---

## Expected Dataset Dimensions

The original dataset has:

```text
569 samples
30 features
```

Therefore:

```text
X.shape
```

returns:

```text
(569, 30)
```

After PCA with three components:

```text
X1.shape
```

returns:

```text
(569, 3)
```

This demonstrates the dimensionality reduction:

```text
569 × 30
    ↓
PCA
    ↓
569 × 3
```

---

## Example Output

The terminal will display information similar to:

```text
Original dataset shape:
(569, 30)

Number of features:
30

Number of samples:
569

Feature names:
['mean radius' 'mean texture' ... ]

PCA-transformed dataset shape:
(569, 3)

Explained variance ratio:
[...]

Total variance explained by 3 components:
...
```

The exact explained variance values are calculated by Scikit-learn from the standardized dataset.

---

## Key Concepts

### 1. Dimensionality Reduction

PCA reduces the number of variables while attempting to preserve important variation in the dataset.

### 2. Principal Components

Principal components are new variables created as linear combinations of the original features.

### 3. Explained Variance

Explained variance indicates how much of the variation in the original data is captured by each principal component.

### 4. Standardization

Standardization places features on a comparable scale before PCA.

### 5. Transformation

After PCA is fitted, the original data can be projected into the principal-component space:

```python
X1 = pca.transform(X_scaled)
```

### 6. Unsupervised Learning

PCA itself does not use the target labels to determine the principal components.

The labels are only used in this project to color the visualization and make the resulting plots easier to interpret.

---

## Original Data vs PCA Data

| Property            |          Original Data |             PCA Data |
| ------------------- | ---------------------: | -------------------: |
| Samples             |                    569 |                  569 |
| Features            |                     30 |                    3 |
| Feature type        |  Original measurements | Principal components |
| Target used for PCA |                     No |                   No |
| Visualization       | Original feature space |         PCA1 vs PCA2 |

The number of samples remains unchanged while the number of dimensions is reduced.

---

## Advantages of PCA

PCA can be useful for:

* Reducing dimensionality
* Visualizing high-dimensional datasets
* Removing redundant information
* Simplifying machine learning datasets
* Reducing computational complexity
* Identifying major patterns in data

However, principal components may be less interpretable than the original features because each component combines information from multiple original variables.

---

## Limitations

PCA also has limitations:

* It creates transformed features that may be difficult to interpret.
* It is sensitive to feature scaling.
* It captures variance rather than class separation.
* Important low-variance information can potentially be discarded.
* PCA assumes that a linear transformation is useful for representing the data.

For these reasons, PCA should be treated as a dimensionality-reduction technique rather than a classification algorithm.

---

## Future Improvements

Possible improvements include:

* Plot all three principal components in 3D.
* Compare PCA with and without feature standardization.
* Calculate cumulative explained variance.
* Determine the number of components required to retain 90%, 95%, or 99% of variance.
* Add a scree plot.
* Compare PCA with other dimensionality-reduction techniques such as t-SNE.
* Use PCA features as input to classification algorithms.
* Compare classification performance before and after dimensionality reduction.
* Save the PCA transformation using Joblib.
* Build a reusable PCA preprocessing pipeline.

---

## License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and build upon this project under the terms of the MIT License.