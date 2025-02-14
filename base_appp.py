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
import joblib,os
from PIL import Image
# Data dependencies
import pandas as pd

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

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
