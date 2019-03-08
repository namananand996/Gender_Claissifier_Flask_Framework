from flask import Flask , request, render_template, url_for
from flask_bootstrap import Bootstrap
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def main():
    return render_template("index.html")



@app.route('/predict',methods=['POST'])
def predict():


    dataset= pd.read_csv("names_dataset.csv")

    # Features and Labels
    X = dataset.iloc[:,1]
    y = dataset.iloc[:,2]
        
    # Vectorization
    countvectorizer = CountVectorizer()
    X = countvectorizer.fit_transform(X) 

    labelEncoder_y = LabelEncoder()
    y = labelEncoder_y.fit_transform(y)


    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25 ,random_state=42 )

    clasifier = MultinomialNB()
    clasifier.fit(X_train,y_train)
    clasifier.score(X_test,y_test)


    # Receives the input query from form
    if request.method == 'POST':
        namequery = request.form['namequery']
        data = [namequery]
        vect = countvectorizer.transform(data).toarray()
        my_prediction = clasifier.predict(vect)

    return render_template("result.html",prediction = my_prediction,name=namequery)




if __name__ == "__main__":
    app.run(debug=True)