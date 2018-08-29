# This is a test function for test functions in pipicat

import numpy as np
from pipicat import cross_val

# Add test
y_pred = [1,2,3,4]
y_true = [1,2,3,5]

#

print(cross_val.my_accuracy_score_regression(y_true,y_pred))

