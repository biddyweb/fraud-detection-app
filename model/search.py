# model building
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# cross-validation
from sklearn.cross_validation import cross_val_score
from sklearn import metrics 
from sklearn.metrics import roc_auc_score

# other
from itertools import product
import numpy as np 


NAMES = { 	LogisticRegression: 'LogisticRegression',
			RandomForestClassifier: 'RandomForestClassifier',
			SVC: 'SVC',
			KNeighborsClassifier: 'KNeighborsClassifier',
			MultinomialNB: 'MultinomialNB'
		}

PIPE = { LogisticRegression:     	{ "penalty": 		["l1", "l2"], 
									  "C": 		 		[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
									  "class_weight": 	[None, "auto"]},
		 RandomForestClassifier: 	{ "n_estimators": 	[10, 100]},
		 # SVC:						{ "C":				[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
		 #							  "kernel":			["rbf", "linear", "poly"],
		 #							  "probability": 	[True]},
		 # KNeighborsClassifier:		{ "n_neighbors": 	[5, 10, 15, 20]}
		 # MultinomialNB:				{ "alpha":			[0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
		}

def run_models(pipe, df, lbl):
	X = df.values
	y = lbl

	for model, param_dict in pipe.iteritems():
		combos = product(*param_dict.values())

		for combo in combos:
			pdict = {k:v for k,v in zip(param_dict.keys(), combo)}
			clf = model(**pdict)
			print NAMES[model], combo
			print np.mean(cross_val_score(clf, X, y, cv=4, scoring="f1"))
			print "-" * 25
