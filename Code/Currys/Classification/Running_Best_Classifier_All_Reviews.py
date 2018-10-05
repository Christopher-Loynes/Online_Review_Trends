## Classification Model - download data and split into training and test sets
import pandas as pd

# Download file to dataframe - this file has already been normaliSed and been through sentiment analysis
# only classifying the category at this point, therefore only care about the Reviews, Delivery and Customer Service labels
chunksize = 10
TextFileReader = pd.read_csv('Copy of SentimentAnalysis_WithSpellcheck.csv', chunksize=chunksize, header=None)
dataset = pd.concat(TextFileReader, ignore_index=False)
dataset.columns = ['Reviews', 'Delivery', 'Customer_Service', 'Purchase_Date', 'Likelihood_to_Recommend',
                   'Overall_Satisfaction','Location', 'Date_Published','Sentiment']

# Remove the first row from the dataset as these have the headers
dataset = dataset.iloc[1:]

# Create an list of all reviews - this will be used to vectorize the words for the independent variable X
corpus = []
for i in range(1, 29779):
    review = dataset['Reviews'][i]
    
    # Remove all capital letters
    review = review.lower()
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer

# Vectorise the model
# Max features set to 5000, due to computational cost
cv = CountVectorizer(max_features = 5000)

# Transform the vector of words to an array containing values for each review
X = cv.fit_transform(corpus).toarray()

# Set up dependant variable (category labels) - delivery is 0, customer service is 1, 0.5 is 'unknown'
# Labels from Reevoo is 0 - bad, 1 - good, 2 - not available
# If the label is 2, then classify as the other option
# If they are both 1 or both 0, then classify as 'unknown'
# If they are 0 and 1, then classify as the 0 category as we we default to the negative feature
y = []
for i in range(1, 29779):
    # For string values
    if dataset['Delivery'][i] == '2':
        y.append(1)
    elif dataset['Customer_Service'][i] == '2':
        y.append(0)
    elif dataset['Delivery'][i] == '0' and dataset['Customer_Service'][i] == '0':
        y.append(0.5)  
    elif dataset['Delivery'][i] == '1' and dataset['Customer_Service'][i] == '1':
        y.append(0.5)  
    elif dataset['Delivery'][i] == '0' and dataset['Customer_Service'][i] == '1':
        y.append(0)
    elif dataset['Delivery'][i] == '1' and dataset['Customer_Service'][i] == '0':
        y.append(1)
        
    # Repeat for numerical values
    elif dataset['Delivery'][i] == 2:
        y.append(1)
    elif dataset['Customer_Service'][i] == 2:
        y.append(0)
    elif dataset['Delivery'][i] == 0 and dataset['Customer_Service'][i] == 0:
        y.append(0.5)  
    elif dataset['Delivery'][i] == 1 and dataset['Customer_Service'][i] == 1:
        y.append(0.5)  
    elif dataset['Delivery'][i] == 0 and dataset['Customer_Service'][i] == 1:
        y.append(0)
    elif dataset['Delivery'][i] == 1 and dataset['Customer_Service'][i] == 0:
        y.append(1)
        
    # Anything else - i.e. bad text, say 'Needs Review'
    else:
        y.append('Needs Review')
        
# For any y items that are marked as 'Needs Review', return the index - these will need
# to be deleted from both X and Y
        
get_indexes = lambda y, xs: [i for (j, i) in zip(xs, range(len(xs))) if y == j]   
del_idx = get_indexes('Needs Review',y)

# sort the indices in reverse order so that when deleted, it deletes properly
del_idx.sort(reverse = True)

# Delete from X and y any indices that are marked as Needs Review - these contain bad data
import numpy as np
for item in del_idx:
    y = np.delete(y, (item), axis = 0)
    X = np.delete(X, (item), axis = 0)
    
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Use SMOTE to oversample the minority classes
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12, ratio = 'minority')
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

from collections import Counter
class_check = Counter(y_train_res)

### Random Forest Classifier
from sklearn.metrics import confusion_matrix
target_names = ['Customer_Service', 'Delivery']
from sklearn.metrics import f1_score
file = open('All Reviews.txt','w')
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Define a formula that will define the classifier, fit the classifier, predict the test results
# Additionally print the accuracy score, confusion matrix and f1-score 
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

# Apply the classifier formula to the 2 datasets - single words without and with SMOTE
#Print a roc curve plot and save
RF_words, cm, score = RFclassifier(X_train, X_test, y_train, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_words, curves = 'each_class', title = 'ROC Curves - Single Word - Random Forest Classifier')
plt.savefig('RF.png')
plt.close()

RF_SMOTE, cm_wSMOTE, score_wSMOTE = RFclassifier(X_train_res, X_test, y_train_res, y_test)
skplt.metrics.plot_roc_curve(y_test, RF_SMOTE, curves = 'each_class', title = 'ROC Curves - Bigrams - Random Forest Classifier')
plt.savefig('RF_SMOTE.png')
plt.close()

file.close()
