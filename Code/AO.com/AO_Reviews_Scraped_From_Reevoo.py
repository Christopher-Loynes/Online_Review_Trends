from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import time
import csv
import os.path

driver = webdriver.Chrome(executable_path=r'/Users/prateek/Desktop/MSc/MWA/WebScraping/chromedriver')
fileName = "AOReevooReviews.csv"
to_Write = ["Reviews", "Delivery", "Customer Service", "Purchase Date", "LikeliHood To Recommend",
            "Overall Satisfaction", "Location", "Date Published"]
proDelF = []
proDelT = []
proCusT = []
proCusF = []

# Initial write to make the headers in the csv file.
with open(fileName, 'a') as data_file:
	csv_writer = csv.writer(data_file, delimiter = ',')
	csv_writer.writerow(to_Write)
	data_file.close()

pagesWithLess = []

#Loops through the pages depending on the range
for j in range(1,5):

	url = 'https://mark.reevoo.com/reevoomark/embeddable_customer_experience_reviews?ajax=true&amp;page='+str(j)+
	'&amp;paginated=true&amp;stylesheet_version=1.5&amp;trkref=APN'
	driver.get(url)
	time.sleep(1)
	body = driver.find_element_by_tag_name('body')
	time.sleep(1)
	html_source = driver.page_source

	page_soup = soup(html_source, 'html.parser')
	with open(fileName, 'a') as data_file:
		csv_writer = csv.writer(data_file, delimiter = ',')
		pageReviews = page_soup.find_all("p", {"class": "comment"})
		allProblem = page_soup.find_all("div", {"class": "service-review-details"})
		allPurchaseDatesFinal = page_soup.find_all("span", {"class": "date date_delivery"})
		likeliHoodToRecommend = page_soup.find_all("div", {"class": "nps-score-label"})
		overallSatisfaction = page_soup.find_all("div", {"class": "response-text"})
		locationInfo = page_soup.find_all("div", {"class": "review-content"})
		datePublished = page_soup.find_all("span", {"class": "date date_publish"})
		time.sleep(1)

		for i in range(1,15):


			proDelF = []
			proDelT = []
			proCusT = []
			proCusF = []

			#Catches if there are less reviews on a page. For some reason there are 14 sometimes.
			try:

				overallSatisfaction = page_soup.find_all("div", {"class": "review-content"})
				if len(allProblem[i-1])>0 or len(pageReviews[i]) > 0:
					proDelT = allProblem[i-1].div.ul.find_all("li", {"class": "delivery true"})
					proDelF = allProblem[i-1].div.ul.find_all("li", {"class": "delivery false"})
					proCusT = allProblem[i-1].div.ul.find_all("li", {"class": "customer-service true"})
					proCusF = allProblem[i-1].div.ul.find_all("li", {"class": "customer-service false"})

				#For Deliveries. If the label is positive or negative
					if len(proDelT)>0:

						proDel_ToWrite = 1
					elif len(proDelF) > 0 :
						proDel_ToWrite = 0
					else:
						proDel_ToWrite = 2

					#For Customer Service. If the label is positive or negaitve
					if len(proCusT) > 0:
						proCusT_ToWrite = 1
					elif len(proCusF) > 0:
						proCusT_ToWrite = 0
					else:
						proCusT_ToWrite = 2
					datePublished_ToWrite = datePublished[i-1].text
					try:
						overallSatisfaction_ToWrite = overallSatisfaction[i-1].ul.li.div.text

					except(AttributeError):
						overallSatisfaction_ToWrite = ""
						
					#locationToWrite = ""
					try:
						locationToWrite = locationInfo[i-1].hgroup.h5.span.text
					except(AttributeError):
						locationToWrite = ""

					toWrite = [pageReviews[i].text, proDel_ToWrite, proCusT_ToWrite, allPurchaseDatesFinal[i-1].text,
                                                   likeliHoodToRecommend[i-1].span.text, overallSatisfaction_ToWrite, locationToWrite, datePublished_ToWrite]
					csv_writer.writerow(toWrite)
					locationToWrite = ""
			except(AttributeError):
				pagesWithLess.append(j)
	print ("page: " + str(j))
