import pickle
import re
import sqlite3 as sql

import bs4 as bs
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
from nltk.corpus import stopwords


def preprocess(review):

    # 1. Remove HTML tags
    review = bs.BeautifulSoup(review).text

    # 2. Use regex to find emoticons
    emoticons = re.findall("(?::|;|=)(?:-)?(?:\)|\(|D|P)", review)

    # 3. Remove punctuation
    review = re.sub("[^a-zA-Z]", " ", review)

    # 4. Tokenize into words (all lower case)
    review = review.lower().split()

    # 5. Remove stopwords
    eng_stopwords = set(stopwords.words("english"))
    review = [w for w in review if not w in eng_stopwords]

    # 6. Join the review to one sentence
    review = " ".join(review + emoticons)
    # add emoticons to the end

    return review


app = Flask(__name__)
placeholder = [None]


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/data")
def list():
    con = sql.connect("database_reviews.db")
    con.row_factory = sql.Row

    cur = con.cursor()
    cur.execute("SELECT * FROM sentiment")

    rows = cur.fetchall()
    return render_template("data.html", rows=rows)


@app.route("/", methods=["POST"])
@app.route("/home", methods=["POST"])
def submit_review():
    if request.method == "POST":
        text = request.form["text1"]

        cleaner = preprocess
        # Load the vectorizer and model
        vect = pickle.load(open("vectorize.pkl", "rb"))
        model = pickle.load(open("model.pkl", "rb"))

        # Preprocess the text
        clean_text = cleaner(text)
        # Vectorize the text
        v = vect.transform([clean_text])
        # Predict the sentiment
        prediction = model.predict(v)[0]

        # Convert the prediction to a string
        if prediction == 1:
            sent = "Positive"
        else:
            sent = "Negative"

        prob = model.predict_proba(v).max()

        # Save the review and the prediction to the database
        conn = sql.connect("database_reviews.db")
        maxid = conn.execute("SELECT MAX(id) FROM sentiment").fetchall()[0][0]

        if not maxid:
            maxid = 0
        maxid += 1
        conn.execute("INSERT INTO sentiment VALUES (?,?,?,?)", (maxid, text, sent, "-"))
        conn.commit()
        conn.close()

        placeholder[-1] = [sent, prob, text]

        return redirect("/result")


@app.route("/result")
def res():
    return render_template(
        "result.html",
        value1=placeholder[-1][0],
        value2=placeholder[-1][1],
        value3=placeholder[-1][2],
    )


@app.route("/result", methods=["POST"])
def feedback():
    if request.method == "POST":
        feed = request.form["rate"]

        con = sql.connect("database_reviews.db")
        maxid = con.execute("SELECT MAX(id) FROM sentiment").fetchall()[0][0]

        # Update the feedback column
        if maxid:
            con.execute(
                f"""UPDATE sentiment 
                        SET feedback = "{feed}"
                        WHERE id = "{maxid}"  
                        """
            )

        con.commit()
        con.close()

        return render_template("thanks.html")


if __name__ == "__main__":
    app.run()
