# get helper functions and variable names
from _functions import *
from _variables import *

# standard things we need
import pandas as pd 
import numpy as np 

# model building 
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn import metrics 

# to output
import cPickle


def create_model():
	print "reading data in" 
	df = get_df('../data/train.json')
	df = get_rid_of_tos_lock(df)

	print "generating labels"
	df, lbl = make_label(df)

	print "natural language processing"
	tfidf_clf, text_vec = make_tfidf(df, TO_TEXT)
	tfidf_lr_clf = LogisticRegression(penalty='l2', C=10.0, class_weight=None)
	tvec_proba = tfidf_lr_clf.fit(text_vec, lbl).predict_proba(text_vec)[:,1]
	df['tproba'] = tvec_proba

	print "dropping some fields"
	df = drop_straight_up(df, TO_DROP)

	print 'fixing some fields'
	df = fix_missing_values(df, MISSING_VALS)
	df["listed"] = np.where(df['listed'] == 'y', 1, 0)

	print "converting if present"
	df = convert_if_present(df, TO_PRESENCE)

	#print "converting on threshold"
	#df = convert_on_threshold(df, TO_THRESHOLD)
	df = drop_straight_up(df, TO_THRESHOLD)

	print "dummytizing"
	df = dummytize(df, TO_DUMMY)

	return df, lbl, tfidf_clf, tfidf_lr_clf


if __name__ == '__main__':
	df, lbl, tfidf_clf, tfidf_lr_clf = create_model()
	final_clf = LogisticRegression(penalty='l2', C=10.0, class_weight='auto')

	print "performing simple 5-fold CV"
	print np.mean(cross_val_score(final_clf, df.values, lbl, cv=5, scoring="roc_auc"))

	print "fitting final model"
	final_model = final_clf.fit(df.values, lbl)

	print "pickling"
	cPickle.dump((tfidf_clf, tfidf_lr_clf, final_model, df.columns), open("model.pkl", "w"))



