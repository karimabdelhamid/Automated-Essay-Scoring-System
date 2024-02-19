import spacy
import pandas as pd
from textstat.textstat import textstatistics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import fitz  # this is used to read pdf files

# load the english language model for processing essays
nlp = spacy.load('en_core_web_sm')

# this function takes an essay and finds different features like word count, sentence count, etc.
def nlp_features(essay):
    doc = nlp(essay)  # process the essay
    num_sentences = len(list(doc.sents))  # count how many sentences there are
    num_words = len([token for token in doc if token.is_alpha])  # count the words
    num_characters = len(essay)  # count all characters INCLUDING spaces and punctuation
    num_syllables = textstatistics().syllable_count(essay)  # count syllables 
    num_complex_words = len([word for word in essay.split() if textstatistics().syllable_count(word) >= 3])  # count complex words
    avg_sentence_length = num_words / num_sentences  # find average sentence length
    avg_syllables_per_word = num_syllables / num_words  # find average syllables per word
    percentage_complex_words = num_complex_words / num_words  # find percentage of complex words
    features = {
        'num_sentences': num_sentences,
        'num_words': num_words,
        'num_characters': num_characters,
        'num_syllables': num_syllables,
        'num_complex_words': num_complex_words,
        'avg_sentence_length': avg_sentence_length,
        'avg_syllables_per_word': avg_syllables_per_word,
        'percentage_complex_words': percentage_complex_words,
    }
    return features  # return the found features as a dictionary

# here are two example essays to show how the system works
essays = ["This is a simple essay. It demonstrates basic structure.",
          "An advanced essay shows complex structures and vocabulary. It demonstrates depth and coherence."]
scores = [3, 5]  # these are made-up scores for the example essays

# convert the essays into a set of features for training our model
features = pd.DataFrame([nlp_features(essay) for essay in essays])

# use tf-idf to figure out which words are most important in the essays
vectorizer = TfidfVectorizer()
tfidf_features = vectorizer.fit_transform(essays).toarray()
tfidf_df = pd.DataFrame(tfidf_features, columns=vectorizer.get_feature_names_out())

# combine the nlp features with the tf-idf scores for a comprehensive view of each essay
combined_features = pd.concat([features, tfidf_df], axis=1)

# split the data into parts for training and testing
X_train, X_test, y_train, y_test = train_test_split(combined_features, scores, test_size=0.2, random_state=42)

# choose a machine learning model to learn from the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # train the model with the training data

# this function predicts the score of an essay based on the trained model
def predict_essay_score(essay):
    essay_features = nlp_features(essay)  # get the nlp features of the essay
    essay_tfidf = vectorizer.transform([essay]).toarray()  # transform the essay with tf-idf
    essay_df = pd.DataFrame(essay_tfidf, columns=vectorizer.get_feature_names_out())  # make a dataframe with tf-idf features
    combined_essay_features = pd.concat([pd.DataFrame([essay_features]), essay_df], axis=1).fillna(0)  # combine all features
    score = model.predict(combined_essay_features)  # predict the score using the model
    return score[0]  # return the predicted score

# function to read the essay text from a pdf file
def read_essay_from_pdf(pdf_path):
    """Reads and returns the text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)  # open the pdf file
        text = ""
        for page in doc:  # for each page in the pdf
            text += page.get_text()  # add the text to our text variable
        return text  # return the collected text
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")  # if there's an error, show it
        return None

if __name__ == "__main__":
    # clear the console for a clean start
    import os
    os.system('cls' if os.name == 'nt' else 'clear')

    # welcome message for the user
    print("Welcome to the Automated Essay Grading System!\n")
    print("This system evaluates your essay based on structure, vocabulary, and relevance.\n")

    # ask the user for the path to their essay pdf
    pdf_path = input("Please enter the full path to your essay PDF file and press Enter:\n")

    # read the essay text from the pdf
    essay_text = read_essay_from_pdf(pdf_path)

    # check if we successfully got text from the pdf
    if essay_text and essay_text.strip():
        # if yes, predict the score of the essay
        predicted_score = predict_essay_score(essay_text)
        print(f"\nAnalyzing your essay...\n")
        # show the predicted score
        print(f"Predicted Score: {predicted_score:.2f}/5.0")
        print("\nThank you for using the Automated Essay Grading System!")
    else:
        # if something went wrong, tell the user
        print("\nNo essay text was detected or an error occurred. Please ensure the file path is correct and try again.")
