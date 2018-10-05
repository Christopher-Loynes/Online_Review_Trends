from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import time
import csv
import os.path

driver = webdriver.Chrome(executable_path=r'INSERT PATH')
fileName = "CurrysReevooReviews.csv"
to_Write = ["Reviews", "Delivery", "Customer Service", "Purchase Date", "LikeliHood To Recommend",
            "Overall Satisfaction", "Location", "Date Published"]

# Four variables below used to check if a person has given a good or a bad score for customer service or delivery
proDelF = []
proDelT = []
proCusT = []
proCusF = []

# Initial write to make the headers in the csv file.
with open(fileName, 'a') as data_file:
	csv_writer = csv.writer(data_file, delimiter = ',')
	csv_writer.writerow(to_Write)
	data_file.close()

# Just to check the pages that have less than 15 reviews.
pagesWithLess = []

#Loops through the pages depending on the range
for j in range(1,5):
	#Currys
	url = 'https://mark.reevoo.com/reevoomark/embeddable_customer_experience_reviews?ajax=true&amp;page='+str(j)+'&amp;paginated=true&amp;stylesheet_version=1.5&amp;trkref=CYS'
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
		
		#allPurchaseDates = page_soup.find_all("p", {"class": "purchase-date"})
		allPurchaseDatesFinal = page_soup.find_all("span", {"class": "date date_purchase"})
		likeliHoodToRecommend = page_soup.find_all("div", {"class": "nps-score-label"})
		overallSatisfaction = page_soup.find_all("div", {"class": "response-text"})
		locationInfo = page_soup.find_all("div", {"class": "review-content"})
		datePublished = page_soup.find_all("span", {"class": "date date_publish"})
		time.sleep(1)
		for i in range(3,18):
			proDelF = []
			proDelT = []
			proCusT = []
			proCusF = []
			
			# There are 14 reviews on a page at times instead of 15.
			try:
				overallSatisfaction = page_soup.find_all("div", {"class": "review-content"})
				if len(allProblem[i-3])>1:
					proDelT = allProblem[i-3].div.ul.find_all("li", {"class": "delivery true"})
					proDelF = allProblem[i-3].div.ul.find_all("li", {"class": "delivery false"})
					proCusT = allProblem[i-3].div.ul.find_all("li", {"class": "customer-service true"})
					proCusF = allProblem[i-3].div.ul.find_all("li", {"class": "customer-service false"})
					
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
						
					#Sometimes the overall is not there. So this catches that
					datePublished_ToWrite = datePublished[i-3].text
					try:
						overallSatisfaction_ToWrite = overallSatisfaction[i-3].ul.li.div.text

					except(AttributeError):
						overallSatisfaction_ToWrite = ""
					try:
						locationToWrite = locationInfo[i-3].hgroup.h5.span.text
					except(AttributeError):
						locationToWrite = ""

					toWrite = [pageReviews[i].text, proDel_ToWrite, proCusT_ToWrite, allPurchaseDatesFinal[i-3].text,
                                                   likeliHoodToRecommend[i-3].span.text, overallSatisfaction_ToWrite, locationToWrite, datePublished_ToWrite]
					csv_writer.writerow(toWrite)
					locationToWrite = ""
			except(IndexError):
				pagesWithLess.append(j)

			#print(len(allProblem))
	print ("page: " + str(j))
