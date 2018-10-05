# Sentiment analysis and classification of reviews posted by customers of Dixons Carphone and ao.com 

Reviews posted by customers of _Dixons Carphone_ and _ao.com_ on [Reevoo](https://www.reevoo.com/en/) are scraped and labelled as _positive_, _negative_ or _neutral_. The reviews are then categorised as relating to _customer service_, _deliveries_ or _other_. 

# Process:

1) **Scrape reviews**
    - Reviews taken from [Reevoo](https://www.reevoo.com/en/) for _Dixons Carphone_ and _ao.com_
        - Selected as reviews have labels relating to _Customer Service_ and _Delivery_
        - Labels used to determine the effectiveness of the _sentiment analysis_ and _classification_ performed
    
2) **Pre-process reviews**
    - Remove tweets that:
        - Contain no text
        - Are duplicates 
        - Contain only numbers, dates, percentages, a single word, a single character and uninformative combinations of characters
    
3) **Text normalisation**
    - Tokenisation
    - Spell checking
    - Stop words
    
4) **Sentiment analysis**
    - 3 different approaches tested:
      - SentiWordNet
      - Stanford CoreNLP
      - TextBlob
    - Effectiveness of an approach is based on the sentiment predicted, against the labels provided on Reevoo:
        - 0 = user is unhappy
        - 1 = user is satisified
        - 2 = label is missing 
    - Evaluated based on _micro-average F1 scores_ calculated _confusion matrices_
      
5) **Sample Bias Correction**
    - _Synthetic Minority Over-sampling Technique (SMOTE)_ to correct class imbalances
    
6) **Classify reviews**
     - 3 different models tested:
       - Naive Bayes
       - Random Forest
       - Support Vector Classifier
     - Each model tested on unigrams, bigrams and trigrams
     - Evaluated on _micro-average F1 scores_ calculated using _confusion matrices_
     - Labels provided on Reevoo used to determine the effectiveness of a classifier
     - The following rules are used for classification:
       - _**Delivery:**_ _Customer Service_ =2 **OR** _Customer Service_ =1 AND _Delivery_ =0
       - _**Customer Service:**_ _Delivery_ = 2 **OR** _Customer Service_ =0 AND _Delivery_ =1
       - _**Unknown:**_ _Customer Service_ =1 AND _Delivery_ =1 **OR** _Customer Service_ =0 AND _Delivery_ =0

## Final Selection

- _Unigrams_
- _Textblobs_ for _sentiment analysis_
- _Random Forest_ for _text classification_

## Results

- 87% of _delivery_ reviews for Currys are positive
- 37% of _customer services_ reviews Currys are positive
   - Concentrated around London, Birmingham and Liverpool
   - Specifically poor _KnowHow_ service documented on _Reevoo_ 
- Recommended to Currys that they improve their customer services:
   - Specifically in London, Birmingham and Liverpool
   - Train staff to improve their interactions with customers (ao.com)
   - Consider closing down stores in underperforming areas
      - Shortly after, Currys announced they would [close 92 stores](https://www.bbc.co.uk/news/business-44286924)


## Resources

- [1) Code](https://github.com/Christopher-Loynes/Online_Review_Trends/wiki/Code)
- [2) Data](https://github.com/Christopher-Loynes/Online_Review_Trends/wiki/Data)



