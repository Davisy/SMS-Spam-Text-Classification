# import important modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from string import punctuation

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,

)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re #regular expression

import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)


import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)


# load data
data = pd.read_csv("../data/spam.tsv", sep="\t")


# replace ham to 0 and spam to 1
new_data = data.replace({"ham": 0, "spam": 1})

stop_words = stopwords.words('english')


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "I am", text)
    text = re.sub(r"ur", " your ", text)
    text = re.sub(r" nd ", " and ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" tkts ", " tickets ", text)
    text = re.sub(r" c ", " can ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r'http\S+', ' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers
    text = re.sub(r" u ", " you ", text)
    text = text.lower()  # set in lowercase

    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])

    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)

    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)

    # Return a list of words
    return (text)


#clean the data
#clean the dataset
new_data["clean_message"] = new_data["message"].apply(text_cleaning)


# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    new_data["clean_message"].values,
    new_data["label"].values,
    test_size=0.15,
    random_state=0,
    shuffle=True,
    stratify=data["label"],
)


print("x_train type: {}".format(type(X_train)))
# clean and transform

vectorizer = CountVectorizer(lowercase=False)
vectorizer.fit(X_train)
X_train_trans = vectorizer.transform(X_train)
X_text_trans = vectorizer.transform(X_test)



# Create a pipeline combing the preprocessing methods and estimator

spam_classifier = MultinomialNB()


# Train the model with cross validation
scores = cross_val_score(spam_classifier,X_train_trans,y_train,cv=10,verbose=3,n_jobs=-1)


# find the mean of the all scores
print("score mean: {}".format(scores.mean()))



# fine turning model parameters

distribution = {"alpha": [1, 0.1, 0.01, 0.001, 0.0001, 0, 0.2, 0.3]}

grid = RandomizedSearchCV(
    spam_classifier,
    param_distributions=distribution,
    n_jobs=-1,
    cv=5,
    n_iter=20,
    random_state=42,
    return_train_score=True,
    verbose=2,
)


# training with randomized search
grid.fit(X_train_trans, y_train)

# summarize the results of the random parameter search
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)


# Train the model with best parameters

best_classifier = MultinomialNB(alpha=1)


#cross validation
scores = cross_val_score(best_classifier, X_train_trans, y_train, cv=10, verbose=2, n_jobs=-1)

print(scores)
print(scores.mean())


# train the best_classifier 
best_classifier.fit(X_train_trans,y_train)


# plot the comfusion matrix
X_test_trans = vectorizer.transform(X_test)



# predict on the test data
y_pred = best_classifier.predict(X_test_trans)


# check the classification report
print(classification_report(y_test, y_pred))


# check accuracy score
print(accuracy_score(y_test, y_pred))


# check f1_ score
print("f1 score: {}".format(f1_score(y_test, y_pred)))



#save model 
import joblib 

joblib.dump(best_classifier, '../models/spam-detection-model.pkl')


#save transformer 
joblib.dump(vectorizer,'../preprocessing/count-vectorizer.pkl')




