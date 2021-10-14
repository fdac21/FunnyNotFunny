import pandas as pd
import csv
import collections
import os
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

jokeIndicators = ["[audience cheers]","[audience cheering]","[laughs]","[applauding]","[woman cheers]","[man cheers]","[applause and cheering]","[cheers]","[cheers and applause]","[cheering]","[audience cheers]","[audience laughs]","[applause]","[laughter]","[distant chuckling"]

class Joke():
    def __init__(self,setup,punchline):
        self.setup = setup
        self.punchline = punchline
        self.taglines = []
    def addTagLine(self, tagline):
        self.taglines.append(tagline)
    def toString(self):
        data = "----Setup----\n"
        data += self.setup
        data += "----Punchline----\n"
        data += self.punchline
        if(len(self.taglines)>0):
            data += "----Tagline(s)----\n"
            for tagline in self.taglines:
                data += tagline
        return data


def findJokes():
    jokes = []
    transcript = "dave-chappelle-the-closer-transcript.txt"
    with open("Dave_Chappelle/"+transcript,"r") as lines:
        setup = ""
        joke = ""
        for line in lines:
            endOfJoke = False 
            for indicator in jokeIndicators:
                if(indicator in line):
                    endOfJoke = True
            if(not endOfJoke):
                print(line)
                setup += joke
                joke = line
            else:
                jokes.append(Joke(setup,joke))
                joke = ""
                setup = ""
        print("---End of Transcript---")
    with open("Dave_Chappelle/dave-chappelle-the-closer-transcript.jokes","w") as parsedJokes:
        for joke in jokes:
            parsedJokes.write(joke.toString() +"\n")
         

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

#We want to remove the data added to the top of the transcript by scraps from the loft:
def cleanData(text):
    data = ""
    line = ""
    previousChar = ''
    nextChar = ''
    newLineChars = ["\n",".","]"]
    #Are we at the content: no
    throughGarbage = False
    for index in range(len(text)):
        char = text[index]
        if (index < len(text)-1):
            nextChar = text[index+1]
        else:
            nextChar = ''
        line += char
        if((char == '\n') or (char ==']') or (char == "." and nextChar != '”') or (char == '”' and previousChar in [".","?","!"])):
            #Save the transcipt
            if(throughGarbage and not("More:" in line)):
                #print(line)
                data += line+"\n"
            #Unless they change their site layout, this is the last link before the content
            if ("Share on linkedin" in line and (len(data)==0)):
                throughGarbage = True
            #The end of each transcript begins with this
            elif(throughGarbage and ("More:" in line)):
                throughGarbage = False
            line = ""
        previousChar = char
    return data

# define a function to get text/clean/calculate frequency
def get_wf (text):
    # remove html markup
    t = cleanHtml (text).lower()
    
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

def inBrackets(text):
    commands = {}
    command = False
    action= ""
    for line in text:
        if "[" == line:
            command = True
        elif "]" == line:
            command = False
            count = commands.get(action,0)
            commands.update({action:count+1})
            action = ""
        elif command:
            action += line
    return commands


def main():
    #Eventually this will be a command line arguement, for now, a string pointing to a file containing links to a  comedian's standup
    comedian = "Dave_Chappelle.txt"
    #Make a directory for their results, if one doesn't exist already
    current_directory = os.getcwd()
    path = os.path.join(current_directory,comedian.split(".")[0].replace("\n",""))
    try:
        os.mkdir(path)
    except OSError as error:
        #Fail quietly
        pass
    with open(comedian,"r") as listOfTranscripts:
        for transcript in listOfTranscripts:
            #Based off of how scrapes from the loft organizes its transripts, this will be the name of the routine
            resultName = transcript.split("comedy/")[1].replace("\n","").replace("/","")
            #crawling raw dataset
            scraped = requests.get(transcript, headers={'User-agent': 'Mozilla/5.0'}).text
            #print(scraped)
            raw_result = cleanHtml(scraped)
            cleaned_result = cleanData(raw_result)
            #save result into the txt file
            with open(path+"/"+resultName+".txt", "w") as result_clean_file:
            	result_clean_file.write(cleaned_result)
            print(inBrackets(cleaned_result))
#main()
findJokes()
####################
#we can use the wf part to count the number of word frequencies
####################
