XGB Classifier in SciKitLearn with LIME explanations for Binary Classification - Base problem category as per Ready Tensor specifications.

- support vector classification
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- binary classification

This is a Binary Classifier using XGB. Model also includes local explanations with LIME for model interpretability.

The classifier starts by trying to find a boundary between the two different classes of data and aims to maximize the distance between the data points and the boundary.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

Hyperparameter Tuning (HPT) performed on xgboost parameters: n_estimators, eta, gamma and max_depth.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

This Binary Classifier is written using Python as its programming language. Python package xgboost is used to implement the main algorithm, evaluate the model, and preprocess the data. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes three endpoints- /ping for health check, /infer for predictions, /explain for local explanations.
