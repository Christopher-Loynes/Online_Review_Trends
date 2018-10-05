from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
import time
import csv

browser = webdriver.Chrome()
driver = webdriver.Chrome(executable_path=r'/Users/prateek/Desktop/MSc/MWA/WebScraping/chromedriver')
driver.get(u'https://uk.trustpilot.com/review/www.currys.co.uk')


time.sleep(1)
body = driver.find_element_by_tag_name('body')

fileName = "UKTrustPilotWithAllInfo.csv"

to_Write = ["Reviews", "CurrysReply", "Number Of Stars", "Customer Location"]
List = []

# Writes initial file headers
with open(fileName, 'a') as data_file:
    csv_writer = csv.writer(data_file, delimiter = ',')
    csv_writer.writerow(to_Write)
    data_file.close()

# Writes all the reviews in the CSV file
with open(fileName, 'a') as data_file:
    csv_writer = csv.writer(data_file, delimiter = ',')
    
    # Loop over 'i' number of pages
    for i in range(1,9):
        url = "https://uk.trustpilot.com/review/www.currys.co.uk?page=" + str(i)
        driver.get(url)
        body = driver.find_element_by_tag_name('body')
        time.sleep(1)
        html_source = driver.page_source
        page_soup = soup(html_source, 'html.parser')
        ReviewContainer = page_soup.find_all("div", {"class": "review-stack"})

        # To loop over all 21 reviews in one page
        for j in range(0, 20):
            
            Review = ReviewContainer[j].find_all("p", {"class": "review-info__body__text"})
            CurrysReply = ReviewContainer[j].find_all("div", {"class": "company-reply__content__body"})
            OneStar = ReviewContainer[j].find_all("div",{"class":"star-rating count-1 size-medium clearfix"})
            TwoStar = ReviewContainer[j].find_all("div",{"class":"star-rating count-2 size-medium clearfix"})
            ThreeStar = ReviewContainer[j].find_all("div",{"class":"star-rating count-3 size-medium clearfix"})
            FourStar = ReviewContainer[j].find_all("div",{"class":"star-rating count-4 size-medium clearfix"})
            FiveStar = ReviewContainer[j].find_all("div",{"class":"star-rating count-5 size-medium clearfix"})
            ConsumerLocation = ReviewContainer[j].find_all("div", {"class":"consumer-info__details__location"})
            noOfStars = 0

            if (len(OneStar)) > 0:
                noOfStars = 1
            elif (len(TwoStar)) > 0:
                noOfStars = 2
            elif (len(ThreeStar)) > 0:
                noOfStars = 3
            elif (len(FourStar)) > 0:
                noOfStars = 4
            elif (len(FiveStar)) > 0:
                noOfStars = 5
                
            # Reset star lists to empty for next review.
            OneStar = []
            TwoStar = []
            ThreeStar = []
            FourStar = []
            FiveStar = []

            ReviewInfoAll = [str(Review[0].text), CurrysReply, noOfStars, ConsumerLocation]
            text_file = open("Output.txt", "a")
            text_file.write(Review[0].text)
            text_file.close()
            csv_writer.writerow(ReviewInfoAll)
data_file.close()
