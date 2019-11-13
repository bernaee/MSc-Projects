### Multivariate Classification

- The data which is multivariate has two classes: positive and negative. It is assumed that the class likelihoods are Gaussian.
- The classifier calculates the class priors, P(Ci), class likelihoods, p(x|Ci), and the evidence, p(x),
the posteriors, P(Ci|x). The different discriminant functions are calculated using the following covariance matrices: 
	- Quadratic Discriminant (A distinct covariance matrix for each class)
	- Linear Discriminant (A common covariance matrix for the two classes)
	- Naive Bayes' Classifier (A diagonal covariance matrix)
	- Euclidean Distance (Equal variance for the two classes)

### Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Function (LDA)
