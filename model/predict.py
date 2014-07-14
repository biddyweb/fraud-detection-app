# to read in file 
import cPickle
import json

# standard things
import pandas as pd 
import numpy as np

# to preprocess example
from _functions import *
from _variables import *
from pandas.io.json import json_normalize


def predict_one(ex_js, tfidf_clf=None, tfidf_lr_clf=None, final_model=None, cols=None):
	
	print "reading data in"
	df = json_normalize(ex_js)

	if tfidf_clf == None:
		tfidf_clf, tfidf_lr_clf, final_model, cols = cPickle.load(open("model.pkl", "r"))

	if 'acct_type' in df.columns:
		del df['acct_type']

	print "natural language processing"
	df["text"] = ""
	for col in TO_TEXT:
		df["text"] += df[col]
	all_doc_text = df["text"].apply(extract_text)
	text_vec = tfidf_clf.transform(df["text"])
	del df['text']

	df['tproba'] = tfidf_lr_clf.predict_proba(text_vec)[:,1]
	
	print "dropping some fields"
	df = drop_straight_up(df, TO_DROP)

	# do a bunch of preprocessing
	print 'fixing some fields'
	df = fix_missing_values(df, MISSING_VALS)
	df["listed"] = np.where(df['listed'] == 'y', 1, 0)

	print "converting if present"
	df = convert_if_present(df, TO_PRESENCE)

	#print "converting on threshold"
	#df = convert_on_threshold(df, model_TO_THRESHOLD)
	df = drop_straight_up(df, TO_THRESHOLD)

	print "stretching dataframe"
	for col in cols:
		if col not in df.columns:
			df[col] = 0

	print "dummytizing"
	for col in TO_DUMMY:
		if col in df.columns: 
			val = df[col][0]
			if val in df.columns:
				df[val] = 1
			df = df.drop(col, axis=1)
		else:
			pass

	#final_df = pd.DataFrame(data=[df[col] if col in df.columns else 0 for col in cols],
	# 						columns= df.columns)
	
	final_df = df[cols]

	return final_model.predict_proba(final_df)[0, 1]



