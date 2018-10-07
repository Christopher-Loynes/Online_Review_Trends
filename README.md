# Sentiment analysis and classification of reviews posted by customers of Dixons Carphone and ao.com 

Code written as part of a group project for the MSc Business Analytics course. Group members:

- [1) Alix Clare](https://www.linkedin.com/in/alixclare/) 
- [2) Denis Aleksandrowicz](https://www.linkedin.com/in/denis-aleksandrowicz-14438b131/)
- [3) Prateek Bawa](https://www.linkedin.com/in/prateek-bawa-957a13ba/) 

The results were presented back to _Dixons Carphone_ and compared against _ao.com_.

# Overview

Reviews posted by customers of _Dixons Carphone_ and _ao.com_ on [Reevoo](https://www.reevoo.com/en/) are scraped and labelled as _positive_ or _negative_. The reviews are then categorised as relating to _customer service_, _deliveries_ or _other_. During testing, labels provided by users on _Reevoo_ were used to select _Textblob_ for sentiment analysis and a _Random Forest_ text classifier that uses unigrams. For the project, the processed tweets were displayed in Tableau to visually demonstrate the findings to _Dixons Carphone_.

# Process:

1) **Scrape reviews**
    - Reviews taken from _Reevoo_ for _Dixons Carphone_ and _ao.com_
        - _Reevoo_ selected as its reviews have labels relating to _Customer Service_ and _Delivery_
        - Labels are used to determine the effectiveness of the _sentiment analysis_ and _classification_ performed
    
2) **Pre-process reviews**
    - Remove tweets that:
        - Contain no text
        - Are duplicates 
        - Contain only numbers, dates, percentages, a single word, a single character and uninformative combinations of characters
    
3) **Text normalisation**
    - _**Performed:**_
         - _Tokenisation_
         - _Spell checking_
         - _Stop words_
    - _**Explored but not performed:**_
         - _Case folding_
         - _Lemmatisation_
         - _Stemming_
         - _Punctuation_
         - _Slang_
    
4) **Sentiment analysis**
    - 3 different approaches tested:
      - _SentiWordNet_
      - _Stanford CoreNLP_
      - _TextBlob_
    - Effectiveness of an approach is based on the sentiment predicted, against the labels provided on _Reevoo_:
        - 0 = user is unhappy
        - 1 = user is satisfied
        - 2 = label is missing 
    - Different approaches evaluated based on _micro-average F1 scores_ calculated using _confusion matrices_
      
5) **Sample Bias Correction**
    - _Synthetic Minority Over-sampling Technique (SMOTE)_ to correct class imbalances
    
6) **Classify reviews**
     - 3 different models tested:
       - _Naive Bayes_
       - _Random Forest_
       - _Support Vector Classifier_
     - Each model tested on _unigrams_, _bigrams_ and _trigrams_
     - Evaluated on _micro-average F1 scores_ calculated using _confusion matrices_
     - Labels provided on _Reevoo_ used to determine the effectiveness of a classifier
     - The following rules are used for classification:
       - _**Delivery:**_ _Customer Service_ =2 **OR** _Customer Service_ =1 AND _Delivery_ =0
       - _**Customer Service:**_ _Delivery_ = 2 **OR** _Customer Service_ =0 AND _Delivery_ =1
       - _**Unknown:**_ _Customer Service_ =1 AND _Delivery_ =1 **OR** _Customer Service_ =0 AND _Delivery_ =0

## Final Selection

- _Unigrams_
- _Textblob_ for _sentiment analysis_
- _Random Forest_ for _text classification_

## Results

- _**87% of _delivery_ reviews for Currys are positive**_

- _**37% of _customer services_ reviews Currys are positive**_
   - Concentrated around London, Birmingham and Liverpool
   - Specifically, poor _KnowHow_ service documented on _Reevoo_ 
   
- _**Recommended to Currys that they improve their customer services:**_
   - Specifically, in London, Birmingham and Liverpool
   - Train staff to improve their interactions with customers, in particular the _KnowHow_ teams
   - Consider closing down stores in underperforming areas
      - Shortly after, Currys announced they would [close 92 stores](https://www.bbc.co.uk/news/business-44286924)


## Resources

- [1) Code](https://github.com/Christopher-Loynes/Online_Review_Trends/wiki/Code)
- [2) Data](https://github.com/Christopher-Loynes/Online_Review_Trends/wiki/Data)



