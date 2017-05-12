Pricing Model Using K-nearest Neighbors
==

This repo contains a class that implements k-nearest neighbors regression that avoids time leakage. 
## Demo
You can run `python house_price_prediction.py` in the command line to see the demo.

## Prepare to install
+ `pandas`
+ `numpy`
+ `sklearn`
+ `tqdm`

## Questions:

1. What is the performance of the model measured in Median Relative Absolute Error?

	The Relative Median Absolute Error (RMAE) is 0.2128.

2. What would be an appropriate methodology to determine the optimal k?

	We can use grid search to determine the optimal k, which performs cross-validation and choose the optimal k with lowest MRAE. Execute the following commands:

	```
     from sklearn.grid_search import GridSearchCV
     parameters = [{'k': np.arange(1,10)}]
     clf = GridSearchCV(housePricePredictor(), parameters, n_jobs=-1)
     clf.fit(X_train, y_train)
     clf.best_params_
	```

3. Do you notice any spatial or temporal trends in error?
	
	There is no noticable spatial or temporal trends in error.

4. How would you improve this model?
	+ Improve the accuracy:
		+ Adding more features to the model
		+ Using grid search cross-validation to find optimal parameters for the model.
	+ Improve the efficiency: 
		+ Precomputing the distance matrix. 	
5. How would you productionize this model?

	Since our K-nearest Neighbors model is computationally intensive, we need to speed up the program. There are several ways we could try:
	+ Using `cython` to speed up computation
	+ Using `numba` to speed up computation
	+ Using processes in parallel with `ProcessPoolExecutor`
	+ Using Apache Spark framework