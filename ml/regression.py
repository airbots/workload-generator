# logistic regression model
# This is just a skeleton code. Will implement with the data from the dataset.

import struct
import gzip
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import gzip

path = "data/load_pattern.txt/"

def load_data():
    with gzip.open(path, 'rb') as f:
        data = np.loadtxt(f)
        return data

# split data with 80% training and 20% testing using train_test_split
def split_data(data, test_size=0.2):
    x = data[:, :-1]
    y = data[:, -1]
    return train_test_split(x, y, test_size=test_size, random_state=0)

data = load_data()
x_train, x_test, y_train, y_test = split_data(data)

print(f"""
      y train: {len(y_train)}, 
      y test: {len(y_test)},
      x train: {x_train.shape}, 
      x test: {x_test.shape}.
      """)

# Model 
lr_model = LogisticRegression(
	penalty="l2",  # Set to "none" to get a prediction front much closer to the NB model trained previously.
	class_weight=None,
	random_state=0,
	solver="lbfgs",
	multi_class="multinomial",
	max_iter=1000)

lr_model.fit(x_train, y_train)
print(f'Accuracy on training data: {lr_model.score(x_train, y_train):.2%}')
print(f'Accuracy on test data: {lr_model.score(x_test, y_test):.2%}')

# Save model
joblib.dump(lr_model, 'lr_model.joblib')

# Evaluation
y_pred_lr = lr_model.predict(x_test)

cm_lr = confusion_matrix(y_test, y_pred_lr)

cm_display = ConfusionMatrixDisplay(cm_lr).plot()