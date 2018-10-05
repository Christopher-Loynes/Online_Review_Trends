## Importing data and split into training and test set
import pandas as pd

# Download file to dataframe - this file has already been normalized and been
# through sentiment analysis. We are only classifying the category at this point,
# so we only care about the Reviews, Delivery and Customer Service labels
chunksize = 10
TextFileReader = pd.read_csv('ManuallyLabelledData.csv', chunksize=chunksize, header=None,encoding='latin-1')
dataset = pd.concat(TextFileReader, ignore_index=False)
dataset.columns = ['Reviews', 'Delivery', 'Customer_Service', 'Purchase_Date',
                   'Likelihood_to_Recommend','Overall_Satisfaction','Location', 
                   'Date_Published', ' Sentiment', 'Sentiment  1 (positive or negative)', 'Sentiment 2 (positive, negative or neutral)',
                   'Category']
# Remove the first row from the dataset as these have the headers
dataset = dataset.iloc[1:]
del chunksize

# Create an list of all reviews - this will be used to vectorize the words for the independent variable X
corpus = []
for i in range(1, 1001):
    review = dataset['Reviews'][i]
    # Remove all capital letters
    review = review.lower()
    corpus.append(review)
del i
del review

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# Vectorize the model for single words, bigrams, and trigrams
cv_words = CountVectorizer(max_features = 5000)
cv_bigrams = CountVectorizer(ngram_range = (2,2),max_features = 5000)
cv_trigrams = CountVectorizer(ngram_range = (3,3), max_features = 5000)
# Transform the vector of words to an array containing values for each review
X_words = cv_words.fit_transform(corpus).toarray()
X_bigrams = cv_bigrams.fit_transform(corpus).toarray()
X_trigrams = cv_trigrams.fit_transform(corpus).toarray()

# Set up dependant variable (category labels) - manually labelled
# Labels from Reevoo is 0 - bad, 1 - good, 2 - not available
# If the label is 2, then classify as the other option
# If they are both 1 or both 0, then classify as 'unknown'
# If they are 0 and 1, then classify as the 0 category as we we default to the negative feature
y = dataset['Category'].values

# Encoding categorical data
# Encoding the Dependent Variable
y = list(y)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train_words, X_test_words, y_train, y_test = train_test_split(X_words, y, test_size = 0.25, random_state = 0)
X_train_bigrams, X_test_bigrams, y_train, y_test = train_test_split(X_bigrams, y, test_size = 0.25, random_state = 0)
X_train_trigrams, X_test_trigrams, y_train, y_test = train_test_split(X_trigrams, y, test_size = 0.25, random_state = 0)

# Use SMOTE to oversample the minority classes
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12)
X_train_res_words, y_train_words = sm.fit_sample(X_train_words, y_train)
X_train_res_bigrams, y_train_bigrams = sm.fit_sample(X_train_bigrams, y_train)
X_train_res_trigrams, y_train_trigrams = sm.fit_sample(X_train_trigrams, y_train)

# The result of this will be 6 datasets.
    # Training and test sets for testing single words with and without oversampling
    # Training and test sets for testing bigrams with and without oversampling
    # Training and test sets for testing trigrams with and without oversampling
    
# Count the number of occurences in the y_train sets to ensure that the oversampling worked 
from collections import Counter
class_check_woSMOTE = Counter(y_train)
class_check_words = Counter(y_train_words)
class_check_bigrams = Counter(y_train_bigrams)
class_check_trigrams = Counter(y_train_trigrams)

### NAIVE BAYES CLASSIFIER ###
# Import the necessary classes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
file = open('Classification Accuracy on 1000.txt','w')
import scikitplot as skplt
import matplotlib.pyplot as plt

# Define a formula that will define the classifier, fit the classifier, predict the test results
# Additionally print the f1 score, confusion matrix to a file and create a probability prediction for the ROC Curve 
def NBclassifier(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test,y_pred, average = None)
    cm = confusion_matrix(y_test, y_pred)
    file.write(str(score)+'\n')
    file.write(str(cm)+'\n\n\n')
    probs = classifier.predict_proba(X_test)
    #skplt.metrics.plot_roc_curve(y_test, probs, curves = 'each_class', title = 'ROC Curves - Naive Bayes Classifier')
    return probs, cm, score

# Apply the classifier formula to the 6 datasets - including printing a ROC curve 
NB_words, cm1, score1 = NBclassifier(X_train_words, X_test_words, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, NB_words, curves = 'each_class', title = 'ROC Curves - Single Word - Naive Bayes Classifier')
plt.savefig('NB_words.png')
plt.close()
NB_bigrams, cm2, score2 = NBclassifier(X_train_bigrams, X_test_bigrams, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, NB_bigrams, curves = 'each_class', title = 'ROC Curves - Bigrams - Naive Bayes Classifier')
plt.savefig('NB_bigrams.png')
plt.close()
NB_trigrams, cm3, score3 = NBclassifier(X_train_trigrams, X_test_trigrams, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, NB_trigrams, curves = 'each_class', title = 'ROC Curves - Trigrams - Naive Bayes Classifier')
plt.savefig('NB_trigram.png')
plt.close()
NB_words_SMOTE, cm4, score4 = NBclassifier(X_train_res_words, X_test_words, y_train_words, y_test)
skplt.metrics.plot_roc_curve(y_test, NB_words_SMOTE, curves = 'each_class', title = 'ROC Curves - Single Word with Oversampling - Naive Bayes Classifier')
plt.savefig('NB_words_SMOTE.png')
plt.close()
NB_bigrams_SMOTE, cm5, score5 = NBclassifier(X_train_res_bigrams, X_test_bigrams, y_train_bigrams, y_test)
skplt.metrics.plot_roc_curve(y_test, NB_bigrams_SMOTE, curves = 'each_class', title = 'ROC Curves - Bigrams with Oversampling - Naive Bayes Classifier')
plt.savefig('NB_bigrams_SMOTE.png')
plt.close()
NB_trigrams_SMOTE, cm6, score6 = NBclassifier(X_train_res_trigrams, X_test_trigrams, y_train_trigrams, y_test)
skplt.metrics.plot_roc_curve(y_test, NB_trigrams_SMOTE, curves = 'each_class', title = 'ROC Curves - Trigrams with Oversampling - Naive Bayes Classifier')
plt.savefig('NB_trigrams_SMOTE.png')
plt.close()


### RANDOM FOREST CLASSIFIER ###
# Import the necessary classes
from sklearn.ensemble import RandomForestClassifier

# Define a formula that will define the classifier, fit the classifier, predict the test results
# Additionally print the f1 score, confusion matrix to a file and create a probability prediction for the ROC Curve
def RFclassifier(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test,y_pred, average = None)
    cm = confusion_matrix(y_test, y_pred)
    file.write(str(score)+'\n')
    file.write(str(cm)+'\n\n\n')
    probs = classifier.predict_proba(X_test)
    return probs, cm, score

# Apply the classifier formula to the 6 datasets
RF_words, cm7, score7 = RFclassifier(X_train_words, X_test_words, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_words, curves = 'each_class', title = 'ROC Curves - Single Word - Random Forest Classifier')
plt.savefig('RF_words.png')
plt.close()
RF_bigrams, cm8, score8 = RFclassifier(X_train_bigrams, X_test_bigrams, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_bigrams, curves = 'each_class', title = 'ROC Curves - Bigrams - Random Forest Classifier')
plt.savefig('RF_bigrams.png')
plt.close()
RF_trigrams, cm9, score9 = RFclassifier(X_train_trigrams, X_test_trigrams, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_trigrams, curves = 'each_class', title = 'ROC Curves - Trigrams - Random Forest Classifier')
plt.savefig('RF_trigram.png')
plt.close()
RF_words_SMOTE, cm10, score10 = RFclassifier(X_train_res_words, X_test_words, y_train_words, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_words_SMOTE, curves = 'each_class', title = 'ROC Curves - Single Word with Oversampling - Random Forest Classifier')
plt.savefig('RF_words_SMOTE.png')
plt.close()
RF_bigrams_SMOTE, cm11, score11 = RFclassifier(X_train_res_bigrams, X_test_bigrams, y_train_bigrams, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_bigrams_SMOTE, curves = 'each_class', title = 'ROC Curves - Bigrams with Oversampling - Random Forest Classifier')
plt.savefig('RF_bigrams_SMOTE.png')
plt.close()
RF_trigrams_SMOTE, cm12, score12 = RFclassifier(X_train_res_trigrams, X_test_trigrams, y_train_trigrams, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_trigrams_SMOTE, curves = 'each_class', title = 'ROC Curves - Trigrams with Oversampling - Random Forest Classifier')
plt.savefig('RF_trigrams_SMOTE.png')
plt.close()

### SVM CLASSIFIER ###
# Import the necessary classes
from sklearn.svm import SVC

# Define a formula that will define the classifier, fit the classifier, predict the test results
# Additionally print the f1 score, confusion matrix to a file and create a probability prediction for the ROC Curve
def SVclassifier(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel = 'linear', random_state = 0, probability = True)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test,y_pred, average = None)
    cm = confusion_matrix(y_test, y_pred)
    file.write(str(score)+'\n')
    file.write(str(cm)+'\n\n\n')
    probs = classifier.predict_proba(X_test)
    return probs, cm, score

# Apply the classifier formula to the 6 datasets
SVC_words, cm13, score13  = SVclassifier(X_train_words, X_test_words, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, SVC_words, curves = 'each_class', title = 'ROC Curves - Single Word - Support Vector Classifier')
plt.savefig('SVC_words.png')
plt.close()
SVC_bigrams, cm14, score14 = SVclassifier(X_train_bigrams, X_test_bigrams, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, SVC_bigrams, curves = 'each_class', title = 'ROC Curves - Bigrams - Support Vector Classifier')
plt.savefig('SVC_bigrams.png')
plt.close()
SVC_trigrams, cm15, score15 = SVclassifier(X_train_trigrams, X_test_trigrams, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, SVC_trigrams, curves = 'each_class', title = 'ROC Curves - Trigrams - Support Vector Classifier')
plt.savefig('SVC_trigram.png')
plt.close()
SVC_words_SMOTE, cm16, score16 = SVclassifier(X_train_res_words, X_test_words, y_train_words, y_test)
skplt.metrics.plot_roc_curve(y_test, SVC_words_SMOTE, curves = 'each_class', title = 'ROC Curves - Single Word with Oversampling - Support Vector Classifier')
plt.savefig('SVC_words_SMOTE.png')
plt.close()
SVC_bigrams_SMOTE, cm17, score17 = SVclassifier(X_train_res_bigrams, X_test_bigrams, y_train_bigrams, y_test)
skplt.metrics.plot_roc_curve(y_test, SVC_bigrams_SMOTE, curves = 'each_class', title = 'ROC Curves - Bigrams with Oversampling - Support Vector Classifier')
plt.savefig('SVC_bigrams_SMOTE.png')
plt.close()
SVC_trigrams_SMOTE, cm18, score18 = SVclassifier(X_train_res_trigrams, X_test_trigrams, y_train_trigrams, y_test)
skplt.metrics.plot_roc_curve(y_test, SVC_trigrams_SMOTE, curves = 'each_class', title = 'ROC Curves - Trigrams with Oversampling - Support Vector Classifier')
plt.savefig('SVC_trigrams_SMOTE.png')
plt.close()

file.close() 

##### SENTIMENT WORD ANALYSIS SCORES - FOR MANUALLY LABELLED DATA - SET SIZE 1000 ###########
#Load the excel files Sentiword, Stanford, and Textblobs.  These contain 2 columns - 
# one is the score given by the algorithm and the other is a manually labelled review 
# for positive and negative (or neutral) sentiment
# These files were taken directly from output provided by the Sentiment Analysis code
# Python was only used for confusion matrix only
df_senti = pd.read_excel('Sentiword.xlsx')
# Creates a confusion matrix based on the values in each column
cm_senti = confusion_matrix(df_senti['Sentiment'], df_senti['Sentiment 2 (positive, negative or neutral)'])
# Calculate a micro average F1-score that provides a score for the whole model
score_senti = f1_score(df_senti['Sentiment'],df_senti['Sentiment 2 (positive, negative or neutral)'], average = 'micro')

df_stan = pd.read_excel('Stanford.xlsx')
cm_stanford = confusion_matrix(df_stan['Sentiment'], df_stan['Sentiment 2 (positive, negative or neutral)'])
score_stanford = f1_score(df_stan['Sentiment'],df_stan['Sentiment 2 (positive, negative or neutral)'], average = 'micro')

df_tb = pd.read_excel('Textblobs.xlsx')
cm_tb = confusion_matrix(df_tb['Sentiment'], df_tb['Sentiment  1 (positive or negative)'])
score_tb = f1_score(df_tb['Sentiment'],df_tb['Sentiment  1 (positive or negative)'], average = 'micro')

# Write scores to text file
file = open('F1 Scores Sentiment on 1000.txt','w')
file.write('SentiWord Micro Average F1 Score ' + str(score_senti)+'\n')
file.write(str(cm_senti)+'\n')
file.write('Stanford Micro Average F1 Score ' + str(score_stanford)+'\n')
file.write(str(cm_stanford)+'\n')
file.write('Textblobs Micro Average F1 Score ' + str(score_tb)+'\n')
file.write(str(cm_tb)+'\n')
file.close()

