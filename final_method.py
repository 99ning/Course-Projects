import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from itertools import chain
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# read in data 

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
## important fields
mask = ["num-comments","prev-games","ratings-received","fun-average",
       "innovation-average","theme-average","graphics-average","audio-average",
        "humor-average","mood-average","audio-rank","humor-rank","mood-rank",
        "fun-rank","innovation-rank","theme-rank","graphics-rank"]

# data cleaning
train_y = train["label"].copy()
train_X = train[mask].copy()
# data cleaning for test data
test_X = test[mask].copy()

# split training data: 20% as validation data
validation_bound = np.rint(np.floor((4/5)*len(train_X))).astype(int)
valid_X = train_X[:][validation_bound:].copy()
train_X = train_X[:][0:validation_bound].copy()
valid_y = train_y[:][validation_bound:].copy()
train_y = train_y[:][0:validation_bound].copy()
eval_set = [(valid_X,valid_y)]


# model init 
clf = CatBoostClassifier(learning_rate=0.01)
# train model
clf.fit(train_X, train_y,early_stopping_rounds=10, eval_set=eval_set,use_best_model=True,silent=True)
# get predictions on test set
clf_predictions = clf.predict(test_X)
clf_predictions = list(chain.from_iterable(clf_predictions))
clf_predictions = np.array(clf_predictions)

# save it as a csv file including a header
pd.DataFrame({"id":test["id"],"label":clf_predictions}).to_csv("submission.csv",index=None)

## get accuracy and F1 score 

'''
cld_valid_pred = clf.predict(valid_X)
print(accuracy_score(valid_y,cld_valid_pred))
print(f1_score(valid_y,cld_valid_pred,average="weighted"))
'''

# 2 plots for CatBoostClassifier
## prediction distribution and feature importance 

'''
values, bins, bars = plt.hist(clf_predictions,bins=6,range=(0,5))
plt.xlabel("Label")
plt.ylabel("Counts")
plt.title("Prediction Distribution for CatBoost")
plt.bar_label(bars, fontsize=10, color='navy')
plt.show()
for i in range(5):
    print(sum(clf_predictions == i))

clf_importances = clf.feature_importances_
clf_feature_names = train_X.columns
boost_importances1 = pd.Series(clf_importances, index=clf_feature_names)
fig, ax = plt.subplots()
boost_importances1.plot.bar(ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
si = np.argsort(clf_importances)[::-1]
for f in range(train_X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,clf_feature_names[si[f]],clf_importances[si[f]]))
'''