### Implementation of Parametric Classification using Maximum Likelihood Estimation.

- The data which is univariate has two classes: positive and negative. It is assumed that the class likelihoods are Gaussian.
- The classifier calculates the class priors, P(Ci), class likelihoods, p(x|Ci), and the evidence, p(x),
the posteriors, P(Ci|x) and the discriminants gi(x) = logP(x|Ci) + logP(Ci).

- Also, we define three different loss function as the followings:
	- 0/1 loss, no rejection
	- asymmetric loss: 0.5/1, no rejection
	- asymmetric loss with an extra action of rejection with a loss of 0.2: 0.5/1, 0.2

### Implementation of Polynomial Regression using Least Squares Method.

- The polynomial model of order 0 to 9 in one variable.
