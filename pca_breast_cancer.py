import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# Load Breast Cancer data
cancer = datasets.load_breast_cancer()

X = cancer.data
y = cancer.target

print("Original dataset shape:")
print(X.shape)

print("\nNumber of features:")
print(X.shape[1])

print("\nNumber of samples:")
print(X.shape[0])

print("\nFeature names:")
print(cancer.feature_names)


# Plot original data
f = plt.figure(1)

plt.scatter(X[:, 0], X[:, 1], c=y)

plt.xlabel(cancer.feature_names[0])
plt.ylabel(cancer.feature_names[1])
plt.title("Original Breast Cancer Data")

plt.tight_layout()


# Standardize the data before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Perform PCA
pca = decomposition.PCA(n_components=3)

pca.fit(X_scaled)

X1 = pca.transform(X_scaled)


# Display PCA information
print("\nPCA-transformed dataset shape:")
print(X1.shape)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained by 3 components:")
print(np.sum(pca.explained_variance_ratio_))


# Plot PCA data
g = plt.figure(2)

plt.scatter(X1[:, 0], X1[:, 1], c=y)

plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.title("PCA Breast Cancer Data")

plt.tight_layout()

plt.show()