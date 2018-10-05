## Classification Model - download data and split into training and test sets
import pandas as pd

# Download file to dataframe - this file has already been normalized and been
# through sentiment analysis. We are only classifying the category at this point,
# so we only care about the Reviews, Delivery and Customer Service labels
chunksize = 10
TextFileReader = pd.read_csv('FullSet.csv', chunksize=chunksize, header=None,encoding='latin-1')
dataset = pd.concat(TextFileReader, ignore_index=False)
dataset.columns = ['Reviews', 'Delivery', 'Customer_Service', 'Purchase_Date', 'Likelihood_to_Recommend','Overall_Satisfaction','Location', 'Date_Published', 'Sentiment']
# Remove the first row from the dataset as these have the headers
dataset = dataset.iloc[1:]

# Create an list of all reviews - this will be used to vectorize the words for the independent variable X
corpus = []
for i in range(1, 25683):
    review = dataset['Reviews'][i]
    # Remove all capital letters
    review = review.lower()
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# Vectorize the model... ### We need to consider if we want to have a max number of words
cv = CountVectorizer(max_features = 5000)
# Transform the vector of words to an array containing values for each review
X = cv.fit_transform(corpus).toarray()

# Set up dependant variable (category labels) - delivery is 0, customer service is 1, 0.5 is 'unknown'
# Labels from Reevoo is 0 - bad, 1 - good, 2 - not available
# If the label is 2, then classify as the other option
# If they are both 1 or both 0, then classify as 'unknown'
# If they are 0 and 1, then classify as the 0 category as we we default to the negative feature
y = []
for i in range(1, 25683):
    # For string values
    if dataset['Delivery'][i] == '2':
        y.append(1)
    elif dataset['Customer_Service'][i] == '2':
        y.append(0)
    elif dataset['Delivery'][i] == '0' and dataset['Customer_Service'][i] == '0':
        y.append(0.5)  ## flaw in this as we had to choose one
    elif dataset['Delivery'][i] == '1' and dataset['Customer_Service'][i] == '1':
        y.append(0.5)  ## flaw in this as we had to choose one
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
        y.append(0.5)  ## flaw in this as we had to choose one
    elif dataset['Delivery'][i] == 1 and dataset['Customer_Service'][i] == 1:
        y.append(0.5)  ## flaw in this as we had to choose one
    elif dataset['Delivery'][i] == 0 and dataset['Customer_Service'][i] == 1:
        y.append(0)
    elif dataset['Delivery'][i] == 1 and dataset['Customer_Service'][i] == 0:
        y.append(1)
    # Anything else - i.e. bad text, say needs review
    else:
        y.append('Needs Review')
        
# For any y items that are marked as Needs Review, return the index - these will need 
        # to be deleted from both X and y
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

# Use ADASYN to oversample the minority classes - this is too computationally expensive
'''from imblearn.over_sampling import ADASYN
ada = ADASYN(random_state=42)
X_ada, y_ada = ada.fit_sample(X_train, y_train)'''

### Random Forest Classifier
from sklearn.metrics import confusion_matrix
target_names = ['Customer_Service', 'Delivery']
from sklearn.metrics import f1_score
file = open('F1 and confusion matrix - All_Reviews_Currys_Classification.txt','w')
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Define a formula that will define the classifier, fit the classifier, predict the test results
# Additionally print the accuracy score, confusion matrix and f1-score 
def RFclassifier(X_train, X_test, y_train, y_test, X):
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    score = f1_score(y_test,y_pred, average = None)
    score_all = f1_score(y_test,y_pred, average = 'micro')
    cm = confusion_matrix(y_test, y_pred)
    file.write('** New Classifier ***')
    file.write('F1 scores by class: '+str(score)+'\n')
    file.write('Confusion Matrix: \n'+str(cm)+'\n\n')
    file.write('F1 Micro Average Score:' +str(score_all)+'\n\n\n')
    probs = classifier.predict_proba(X_test)
    y_pred_all = classifier.predict(X)
    return probs, cm, score, y_pred, y_pred_all, score_all

# Apply the classifier formula to the 2 datasets - single words without and with SMOTE
#Print a roc curve plot and save
RF_words, cm, score, y_pred, y_pred_all, score_all = RFclassifier(X_train, X_test, y_train, y_test, X)
skplt.metrics.plot_roc_curve(y_test, RF_words, curves = 'each_class', title = 'ROC Curves - Single Word - Random Forest Classifier')
plt.savefig('RF.png')
plt.close()
RF_SMOTE, cm_wSMOTE, score_wSMOTE, y_pred_SMOTE, y_pred_SMOTE_all, score_all_SMOTE = RFclassifier(X_train_res, X_test, y_train_res, y_test, X)
skplt.metrics.plot_roc_curve(y_test, RF_SMOTE, curves = 'each_class', title = 'ROC Curves - Single Word using SMOTE - Random Forest Classifier')
plt.savefig('RF_SMOTE.png')
plt.close()

file.close()

# Concatenate the predictions of X - this is going to be combined with the output so
# we can have a predicted category for each review and compare with the sentiment
y_pred_all = list(y_pred_all)
y_pred_SMOTE_all = list(y_pred_SMOTE_all)
preds = pd.DataFrame({'Prediction': y_pred_all, 'Prediction with SMOTE': y_pred_SMOTE_all})

# Delete the lines that had bad data and Re-index the dataset sub to line up with the predictions   
dataset_sub = dataset.drop(dataset.index[del_idx])
dataset_sub.index = range(len(dataset_sub))

# Export the predictions to excel    
fullset = pd.concat([dataset_sub, preds], axis = 1, ignore_index = True)
fullset.to_excel('FullSet With Predictions.xlsx', sheet_name='sheet1', index=False)


##### SENTIMENT WORD ANALYSIS SCORE - TEXTBLOBS ONLY ON FULL SET ###########
#Pull the two columns that contain the scores for Textblobs  These are two columns in dataset - 
# one is the score given by the algorithm and the other is a manually labelled review 
# for positive and negative (or neutral) sentiment
# These results were taken directly from output provided by the Sentiment Analysis code
# Python was only used for confusion matrix only
# Creates a confusion matrix based on the values in each column

# Convert the column Likelihood to Recommend to a binary category 0 and 1 
ltr_conversion = []
for i in range(0,25076):
    dataset_sub['Likelihood_to_Recommend'][i] = float(dataset_sub['Likelihood_to_Recommend'][i])
    if dataset_sub['Likelihood_to_Recommend'][i] <0.5:
        ltr_conversion.append(0)
    elif dataset_sub['Likelihood_to_Recommend'][i] >=0.5:
        ltr_conversion.append(1)
    else:
        ltr_conversion.append('Needs Review')

# Pull the column sentiment - this is from textblobs and has pos and neg values, so encode to 0 and 1        
tb = list(dataset_sub['Sentiment'])
from sklearn.preprocessing import LabelEncoder
labelencoder_tb = LabelEncoder()
tb = labelencoder_tb.fit_transform(tb)

# create a confusion matrix of all    
cm_tb_all = confusion_matrix(ltr_conversion, tb)
# Calculate a micro average F1-score that provides a score for the whole model
score_tb_all = f1_score(ltr_conversion, tb, average = 'micro')

# write to a text file
file = open('F1 Scores Sentiment on Fullset.txt','w')
file.write('Textblobs Micro Average F1 Score ' + str(score_tb_all)+'\n')
file.write(str(cm_tb_all)+'\n')
file.close() 
