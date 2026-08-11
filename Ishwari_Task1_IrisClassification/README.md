Iris Flower Classification

This project builds a machine learning model to classify iris flowers into three species (setosa, versicolor, virginica) based on their sepal and petal measurements. Built as part of the Oasis Infobyte Data Science Internship.

Dataset
The classic Iris dataset, loaded directly from scikit-learn (no external download needed). Contains 150 samples, 50 of each species, with four measurements per flower.

Approach
Exploratory data analysis with pairplots and box plots to understand feature distributions across species. Three classification models were built using scikit-learn Pipelines (which combine feature scaling and the classifier into a single step): Logistic Regression, K-Nearest Neighbors, and Random Forest. Each model was evaluated using 5-fold cross-validation for a reliable