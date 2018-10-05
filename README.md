# Sentiment analysis and classification of reviews posted by customers of Dixons Carphone and ao.com 

Reviews posted by customers of _Dixons Carphone_ and _ao.com_ on [Reevoo](https://www.reevoo.com/en/) are scraped and labelled as _positive_, _negative_ or _neutral_. The reviews are then categorised as relating to _customer service_, _deliveries_ or _other_. 

Process completed:

1) **Scrape reviews**
    - Reviews taken from [Reevoo](https://www.reevoo.com/en/) for _Dixons Carphone_ and _ao.com_
2) **Pre-process reviews**
    - Reviews contained no text
    - Duplicate reviews
    - Contained only numbers, dates, percentages, a single word, a single character and uninformative combinations of character
3) **Text normalisation**
    - Tokenisation
    - Spell checking
    - Stop words
4) **Sentiment analysis**
    - 3 different approaches tested:
      - SentiWordNet
      - Stanford CoreNLP
      - TextBlob
5) **Sample Bias Correction**
  - _Synthetic Minority Over-sampling Technique (SMOTE)_ to correct class imbalances
6) **Classify reviews**
     - 3 different models tested:
       - Naive Bayes
       - Random Forest
       - Support Vector Classifier

## Results

- 87% of _delivery_ reviews for Currys are positive
- 37% of _customer services_ reviews Currys are positive
   - Concentrated around London, Birmingham and London
     - Specifically poor _KnowHow_ service documented on _Reevoo_ 

## Resources

- [1) Code]()
- [2) Data]()



