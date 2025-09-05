# Machine Learning Project: Supervised and Unsupervised Learning Implementation

## Overview

This project implements a comprehensive machine learning pipeline that includes both supervised and unsupervised learning algorithms from scratch. The project demonstrates end-to-end machine learning workflow including data gathering, exploratory data analysis (EDA), data preprocessing, model implementation, validation, and evaluation.

### Pipeline Components

1. **Data Gathering & EDA**: Collection and exploration of structured datasets with comprehensive statistical analysis
2. **Data Preprocessing**: Data cleaning, transformation, feature selection, and preparation for modeling
3. **Supervised Learning**: Implementation of 6 core algorithms (KNN, Logistic/Polynomial Regression, Gaussian Naive Bayes, CART, SVM, ANN)
4. **Unsupervised Learning**: Implementation of 3 clustering and dimensionality reduction algorithms (K-means, DBSCAN, PCA)
5. **Model Validation**: Both hold-out validation and k-fold cross-validation with performance comparison

### Key Features

- **From-scratch implementations** using only mathematical libraries (NumPy, Pandas, SciPy)
- **Library implementations** using scikit-learn, PyTorch, or TensorFlow for comparison
- **Comprehensive validation strategies** to prevent data leakage
- **Performance analysis** and improvement recommendations
- **Modular design** for easy maintenance and extension

## How to Run

### Supervised Learning

1. **Start with EDA**: Run all EDA cells sequentially to understand the dataset
2. **Data Preprocessing**: Always run the "Data Splitting" cell first
3. **Validation-specific cells**: Some cells are only needed for hold-out validation (marked in markdown)
4. **Algorithm-specific workflow**:
   - Import required modules (under "Pre: [Algorithm Name]")
   - Initialize based on validation type (under "Pre: [Algorithm] - [Validation Type]")
   - Run setup helper functions in "Modeling and Validation" section
   - Split training data into k-folds for cross-validation
   - Execute both from-scratch and scikit-learn implementations

**Example for K-Fold KNN**:
1. Run code under "Pre: K-Nearest Neighbors (KNN)"
2. Run code under "Pre: KNN - K-Fold Cross-Validation" 
3. Run setup helper functions and k-fold splitting in "Modeling and Validation"
4. Execute K-Fold cells for both custom and scikit-learn implementations

### Unsupervised Learning

1. **Algorithm-specific execution**: Each algorithm has its own markdown section
2. **General workflow**:
   - Run code under "Unsupervised Learning" markdown
   - Execute algorithm-specific cells sequentially
   - Compare from-scratch vs library implementations

## Implemented Algorithms

### Supervised Learning (Section 2)
- [x] **K-Nearest Neighbors (KNN)**: Multiple distance metrics, configurable neighbors
- [x] **Logistic/Softmax/Polynomial Regression**: Gradient descent optimization, regularization
- [x] **Gaussian Naive Bayes**: For classification tasks
- [x] **CART (Decision Trees)**: Both classification and regression variants
- [x] **Support Vector Machine (SVM)**: Linear and non-linear kernels
- [x] **Artificial Neural Network (ANN)**: Feedforward networks with multiple activation functions

**Bonus Features Implemented:**
- [x] **LogReg**: Newton's method
- [x] **SVM**: RBF Kernel
- [x] **ANN**: Adam Optimizer, Xavier, He Initializer
- [x] **Ensemble Learning**: Random Forest

### Unsupervised Learning (Section 3)
- [x] **K-means Clustering**: Random initialization with k-means++ bonus
- [x] **DBSCAN**: Density-based clustering with noise handling  
- [x] **Principal Component Analysis (PCA)**: Dimensionality reduction with explained variance

**Bonus Features Implemented:**
- [x] **K-MEANS++**: Improved initialization method
