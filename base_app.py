"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib, os
from PIL import Image

# Data dependencies
import pandas as pd
import numpy as np

# Data plots
from textblob import TextBlob

# from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt

# data manipulation
import string
import regex as re

# word tokenize
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
import nltk
from nltk.corpus import stopwords


# Vectorizer
news_vectorizer = open("resources/vectorizer_3.pkl", "rb")
tweet_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# hash tags negative class
raw = pd.read_csv("resources/train.csv")

# Load Anti class data
anti_hash = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_anti_hashtags.csv"
)
anti_retweet = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_Anti_retweets.csv"
)

# Load Anti class data
neutral_hash = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_Neutral_hashtags.csv"
)
neutral_retweet = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_Neutral_retweet.csv"
)

# Load Anti class data
pro_hash = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_Pro_hashtags.csv"
)
pro_retweet = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_Pro_retweets.csv"
)

# Load Anti class data
news_hash = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_News_hashtags.csv"
)
news_retweet = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_News_retweets.csv"
)

# Load word count dataframe
token_df = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/df_token_ind.csv"
)

# getting the analysis data frame
clean_df = pd.read_csv(
    "https://github.com/kabirodavies/streamlit/blob/master/resources/data/clean_df.csv"
)


def plot_target_based_features(feature):
    
    fig, ax = plt.subplots(1, 1, "all", figsize=(15, 9))
    # create data frames for classes
    x1 = clean_df[clean_df["sentiment"] == 1][feature]
    x2 = clean_df[clean_df["sentiment"] == 2][feature]
    x0 = clean_df[clean_df["sentiment"] == 0][feature]
    x_neg = clean_df[clean_df["sentiment"] == -1][feature]
    # plt.figure(1, figsize = (16, 8))
    plt.xlabel("number of charecters in a tweet")
    plt.ylabel("number of tweets")
    plt.title("Word distribution chart")

    _ = plt.hist(x1, alpha=0.5, color="grey", bins=50, label="belivers")
    _ = plt.hist(x2, alpha=0.5, color="blue", bins=50, label="news")
    _ = plt.hist(x0, alpha=0.6, color="green", bins=50, label="neutral")
    _ = plt.hist(x_neg, alpha=0.5, color="orange", bins=50, label="anti")
    plt.legend(["belivers", "news", "neutral", "anti"])

    st.pyplot(fig)

    return _


# Making sentences for word  cloud retweets
tweet_pro = " ".join(
    [review for review in pro_retweet.Retweets if review is not np.nan]
)
tweet_neutral = " ".join(
    review for review in neutral_retweet.Retweets if review is not np.nan
)
tweet_news = " ".join(
    review for review in news_retweet.Retweets if review is not np.nan
)
tweet_anti = " ".join(
    review for review in anti_retweet.Retweets if review is not np.nan
)

# Making sentences for word cloud hashtags
hash_pro = " ".join([review for review in pro_hash.hashtags if review is not np.nan])
hash_neutral = " ".join(
    review for review in neutral_hash.hashtags if review is not np.nan
)
hash_news = " ".join(review for review in news_hash.hashtags if review is not np.nan)
hash_anti = " ".join(review for review in anti_hash.hashtags if review is not np.nan)



# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "EDA", "Project Team", "Hyperparamater Tuning"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("This is a simple Streamlit app that will help you build a classification model to help tackle global warming.")
		st.write('Global warming is a growing issue that affects the Earth’s climate. ')	
		image= Image.open('resources/imgs/climate.jpeg')
		st.image(image, caption='https://www.un.org/en/climatechange/what-is-climate-change', use_column_width=True)
		# You can read a markdown file from supporting resources folder
		st.markdown("Climate change refers to long-term shifts in temperatures and weather patterns.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with Multinomial Logistic Regression with TF-IDF Vectorizer and GridSearchCV")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logisticregmulti.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Text Categorized as: {}".format(prediction))
	if selection == "EDA":
		st.info("We removed issues which may impact data modelability")
		# You can read a markdown file from supporting resources folder
		st.markdown("Investigated - visualisation, summarised, cleaned")
	if selection == "Project Team":
		st.info("Hassan Juma & Davis Njogu")
		# You can read a markdown file from supporting resources folder
		st.markdown("Thanks it was nice working on this project")
	if selection == "Hyperparamater Tuning":
		st.info("Hyperparameter tuning is choosing a set of optimal hyperparameters for a learning algorithm. ")
		# You can read a markdown file from supporting resources folder
		st.markdown("A hyperparameter is a model argument whose value is set before the learning process begins. The key to machine learning algorithms is hyperparameter tuning.")


# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
