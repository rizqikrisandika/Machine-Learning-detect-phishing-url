<<<<<<< HEAD
import numpy as np
import pandas as pd
import feature_extraction
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from flask import jsonify


def getResult(url):

    #Importing dataset
	
    df = pd.read_csv("dataset_v3.csv")

    #Seperating features and labels
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    #Seperating training features, testing features, training labels & testing labels
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
    bg = BaggingClassifier(DecisionTreeClassifier(), 
        n_estimators=100,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        random_state=27,
        verbose=0
    )
    bg.fit(x_train,y_train)
    score = bg.score(x_test,y_test)
	
    print('Akurasi : ',score*100)

    X_new = []

    X_input = url
    X_new=feature_extraction.generate_data_set(X_input)
    X_new = np.array(X_new).reshape(1,-1)

    try:
        prediction = bg.predict(X_new)
        if prediction == 1:
            return "Bukan Website Phishing"
        else:
            return "Terindikasi Website Phishing"
    except:
        return "Terindikasi Website Phishing"
=======
import numpy as np
import pandas as pd
import feature_extraction
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from flask import jsonify


def getResult(url):

    #Importing dataset
	
    df = pd.read_csv("dataset_v3.csv")

    #Seperating features and labels
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    #Seperating training features, testing features, training labels & testing labels
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=7)
    bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, n_estimators=100)
    bg.fit(x_train,y_train)
    score = bg.score(x_test,y_test)
	
    print('Akurasi : ',score*100)

    X_new = []

    X_input = url
    X_new=feature_extraction.generate_data_set(X_input)
    X_new = np.array(X_new).reshape(1,-1)

    try:
        prediction = bg.predict(X_new)
        if prediction == 1:
            return "Bukan Website Phishing"
        else:
            return "Terindikasi Website Phishing"
    except:
        return "Terindikasi Website Phishing"
>>>>>>> add112012562178373980aeafff718f900fe4b31
