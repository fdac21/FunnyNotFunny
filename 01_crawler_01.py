import pandas as pd
import csv
import collections
import requests, re, nltk
from bs4 import BeautifulSoup
from nltk import clean_html
from collections import Counter
import operator

nltk.download('stopwords')

# we may not care about the usage of stop words
stop_words = nltk.corpus.stopwords.words('english') + [ 'ut', '\'re','.', ',', '--', '\'s', '?', ')', '(', ':', '\'', '\"', '-', '}', '{', '&', '|', u'\u2014', '' ]

# We most likely would like to remove html markup
# crawl elemenets under div > p
def cleanHtml (html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    for p in soup.find_all('p'):
    #soup = soup.find_all('div.elementor-widget-container')
        return soup .get_text()

# We also want to remove special characters, quotes, etc. from each word
def cleanWord (w):
    # r in r'[.,"\']' tells to treat \ as a regular character 
    # but we need to escape ' with \'
    # any character between the brackets [] is to be removed 
    wn = re.sub('[,"\.\'&\|:@>*;/=]', "", w)
    # get rid of numbers
    return re.sub('^[0-9\.]*$', "", wn)
       
# define a function to get text/clean/calculate frequency
def get_wf (text):
    # remove html markup
    t = cleanHtml (text) .lower()
    
    # split string into an array of words using any sequence of spaces "\s+" 
    wds = re .split('\s+',t)
    
    # remove periods, commas, etc stuck to the edges of words
    for i in range(len(wds)):
        wds [i] = cleanWord (wds [i])
    
    # If satisfied with results, lets go to the next step: calculate frequencies
    # We can write a loop to create a dictionary, but 
    # there is a special function for everything in python
    # in particular for counting frequencies (like function table() in R)
    wf = Counter (wds)
    
    # Remove stop words from the dictionary wf
    for k in stop_words:
        wf. pop(k, None)
        
    #how many regular words in the document?
    tw = 0
    for w in wf:
       tw += wf[w] 
    
    # Get ordered list
    wfs = sorted (wf .items(), key = operator.itemgetter(1), reverse=True)
    ml = min(len(wfs),15)

    #Reverse the list because barh plots items from the bottom
    return (wfs [ 0:ml ] [::-1], tw)
        

#crawling raw dataset
raw_result = cleanHtml(requests.get('https://scrapsfromtheloft.com/comedy/dave-chappelle-the-closer-transcript/', headers={'User-agent': 'Mozilla/5.0'}).text)

#save result into the txt file
with open("02_result_raw_data.txt", "w") as result_raw_file:
	result_raw_file.write(raw_result)
print(raw_result)



####################
#we can use the wf part to count the number of word frequencies
####################
