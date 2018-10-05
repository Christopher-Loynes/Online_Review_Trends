from pycorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from textblob import Word
from textblob.sentiments import NaiveBayesAnalyzer
from textblob import TextBlob
import csv
import time
from nltk.stem import WordNetLemmatizer
nlp = StanfordCoreNLP('http://localhost:9000')

# Function to write the headers in the CSV
def InitialWriteToFile():

	to_Write = ["Reviews", "Delivery", "Customer Service", "Purchase Date", "LikeliHood To Recommend",
                    "Overall Satisfaction", "Location", "Date Published", "Sentiment"]
	fileN = "TestToSentimentAnalyse_TextBlobsWithSpellCheck_TwentySevenManually.csv"
	with open(fileN, 'a') as data_file:
		csv_writer = csv.writer(data_file, delimiter = ',')
		csv_writer.writerow(to_Write)
		data_file.close()

# Function to write the reviews and all its information.
# Writes in the order of the parameters
def FinalWrite(Review, Deliver, CustomerService, purchaseDate, LikeliHood, Overall,location, datePublished, Sentiment):

	fileN = "TestToSentimentAnalyse_TextBlobsWithSpellCheck_TwentySevenManually.csv"
	to_Write = [str(Review), str(Deliver), str(CustomerService), str(purchaseDate), str(LikeliHood), str(Overall),
                    str(location), str(datePublished), str(Sentiment)]
	with open(fileN, 'a') as data_file:
		csv_writer = csv.writer(data_file, delimiter = ',')
		csv_writer.writerow(to_Write)
		data_file.close()


# Function that normalises all the reviews.
# 1. Tokenises
# 2. SpellChecks
# 3. Skips if a review does not contain a NP or a VB
# 4. Skips if length of review is less than 2 in size
# This function in turn calls textblobs
def AnalyseSentiments():

	nlp = StanfordCoreNLP('http://localhost:9000')
	Reviews = []
	operations = {'annotators': 'tokenize,lemma,pos,sentiment',
             'outputFormat': 'json'}
	allReviewData = []
	InitialWriteToFile()
	l = -1
	fileName = "TwentySeven.csv"
	with open(fileName, 'r') as f:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			l += 1
			try:
				if l == 28:	#Break at the last row. Avoids errors
					break
				tokens = nltk.word_tokenize(row[0])
				Reviews.append(row[0])
				if not len(tokens)<2:
					allReviewData.append(row)
				print(str(l))
			except(IndexError):
				pass
	filtered_Reviews = []

	print("Finished reading")
	for Review in Reviews:
		tokens = nltk.word_tokenize(Review)
		if not len(tokens) < 2:
			filtered_Reviews.append(tokens)
	allReviewData_Final = allReviewData
	onReviewNumber = 0
	ReviewsJoinedToMakeASentence = ""
	filtered_ReviewsAfterProcess1 = []
	print("Finished Filtering Reviews with len less than 2\n")
	print("Starting spell check\n")
	t0 = time.time()
	spellCount = 0
	for Review in filtered_Reviews:
		JoinedTokens = ' '.join(word for word in Review)
		filtered_review = TextBlob(str(JoinedTokens))
		JoinedTokens = str(filtered_review.correct())
		filtered_ReviewsAfterProcess1.append(JoinedTokens)
		spellCount += 1
		if spellCount%50 == 0:
			print("Spellchecked: "+str(spellCount))

	flag = 0
	i = 0
	pos  = []
	reviewsWithSentiment = []
	print("Starting Sentiment Analysis")
	counter = -1
	for Review in filtered_ReviewsAfterProcess1:
		res = nlp.annotate(Review,operations)
		counter += 1
		try:
			for s in res["sentences"]:
				for token in s["tokens"]:
					stringNone = str(token["pos"])
					pos.append(token["pos"])
					# Flags if a review has a verb or noun.
					if str(token["pos"]) == "NN" or str(token["pos"]) == "NNS" or str(token["pos"]) == "NNP" or
					str(token["pos"]) == "NNPS":
						flag = 1
					if str(token["pos"]) == "VB" or str(token["pos"]) == "VBG" or str(token["pos"]) == "VBD" or
					str(token["pos"]) == "VBN" or str(token["pos"]) == "VBP" or str(token["pos"]) == "VBZ":
						flag = 1
			if flag == 1:
				pos = []
				flag = 0
				reviewWithoutStopWords, classification = textBllob(Review)
				if counter %50 == 0:
					print ("Sentiment Analysed: "+str(counter))
				try:
					FinalWrite(str(reviewWithoutStopWords), str(allReviewData_Final[counter][1]),
                                                   str(allReviewData_Final[counter][2]), str(allReviewData_Final[counter][3]),
                                                   str(allReviewData_Final[counter][4]), str(allReviewData_Final[counter][5]),
                                                   str(allReviewData_Final[counter][6]), str(allReviewData_Final[counter][7]), str(classification))
					time.sleep(1)
				except(IndexError):
					pass
		except(TypeError):
			pass

# Python third party package
def textBllob(text):
	stop_words = set(stopwords.words('english'))

	stopwordsRemoved = [token for token in nltk.word_tokenize(text) if not token in stop_words]
	filteredReview = ' '.join(word for word in stopwordsRemoved)

	blob = TextBlob(text, analyzer=NaiveBayesAnalyzer())
	classification, pos, neg = blob.sentiment
	return (filteredReview,classification)


AnalyseSentiments()
