# cp2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from pandas.core.frame import DataFrame
alldata=pd.read_csv("training_data.csv")
data_text=alldata['text']
data_stars=alldata['stars']
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer()
tfidf_vectorizer.fit(data_text)
data_text=tfidf_vectorizer.transform(data_text).toarray()
#==========================================================================
testdata=pd.read_csv("test_data.csv")
test=testdata['text']
test=tfidf_vectorizer.transform(test).toarray()
#=========================================================================
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(data_text,data_stars)
out=clf.predict(test)
#print(out)
outdf=pd.DataFrame(testdata['review_id'])
outdf['stars']=out
outdf.to_csv("cp2_out.csv",header=False, index=False)
r=pd.read_csv('cp2_out.csv')
r
