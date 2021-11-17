import pandas as pd
import csv
import collections
import os
import requests, re, nltk
from bs4 import BeautifulSoup
from nltk import clean_html
from collections import Counter
import operator
import numpy as np
import pylab
import matplotlib.pyplot as plt
import random

nltk.download('stopwords')
def cleanWord2 (w):
    # r in r'[.,"\']' tells to treat \ as a regular character 
    # but we need to escape ' with \'
    # any character between the brackets [] is to be removed 
    wn = re.sub('[…\n.!?"“”‘’\n,]', "", w)
    # get rid of numbers
    return re.sub('^[0-9\.]*$', "", wn)


# we may not care about the usage of stop words
stop_words = nltk.corpus.stopwords.words('english') + [ 'ut', '\'re','.', ',', '--', '\'s', '?', ')', '(', ':', '\'', '\"', '-', '}', '{', '&', '|', u'\u2014', '', " " ]

# We most likely would like to remove html markup
# crawl elemenets under div > p

#Some of these indicators are not for jokes. We need to be selective.
#jokeIndicators = ["[audience cheers]","[audience cheering]","[laughs]","[applauding]","[woman cheers]","[man cheers]","[applause and cheering]","[cheers]","[cheers and applause]","[cheering]","[audience cheers]","[audience laughs]","[applause]","[laughter]","[distant chuckling"]
#jokeIndicators = ["[laughs]","[audience laughs]","[laughter]","[distant chuckling"]
jokeIndicators = ["cheers and applause","audience laughing","laughing","laughs","cheers","audience cheers","audience laughs","applause","laughter","cheering","distant chuckling","Laughter","Laughter and applause","Light laughter","Moans and applause","Moans and laughter","chuckles"]

def wordsToPhrase(data,phraseSize):
    phrases = []
    phrase = []
    for word in data:
        if len(phrase) < phraseSize:
            phrase.append(word)   
        else:
            temp =""
            for part in phrase:
                temp += part + " "
            phrases.append(temp)
            phrase.pop(0)
            phrase.append(word)
            if(word[:-1] in [".","!","?"]):
                temp =""
                for part in phrase:
                    temp += part + " "
                phrases.append(temp)
                phrase = []
    return phrases


class Joke():
    def __init__(self,setup,punchline):
        self.setup = setup
        self.punchline = punchline
        self.taglines = []
    def addTagLine(self, tagline):
        self.taglines.append(tagline)
    def getAllPhrases(self,n):
        wds = self.setup.split(" ")
        for i in range(len(wds)):
            wds[i]= cleanWord2(wds[i]).lower()
        phrases = wordsToPhrase(wds,n)
        wds = self.punchline.split(" ")
        for i in range(len(wds)):
            wds[i]= cleanWord2(wds[i]).lower()
        for phrase in wordsToPhrase(wds,n):
            phrases.append(phrase)

        wds = []
        for line in self.taglines:
            for word in line.split(" "):
                wds.append(word)
        for i in range(len(wds)):
            wds[i]= cleanWord2(wds [i]).lower()
        for phrase in wordsToPhrase(wds,n):
            phrases.append(phrase)
        return phrases
    def setupAnalysis(self):
        wds = self.setup.split(" ")
        for i in range(len(wds)):
            wds[i]= cleanWord2(wds[i]).lower()
        wf = Counter (wds)
        tw = 0
        for w in wf:
            tw += wf[w]
        return wf, tw
    def punchLineAnalysis(self):
        wds = self.punchline.split(" ")
        for i in range(len(wds)):
            wds[i]= cleanWord2(wds[i]).lower()
        wf = Counter (wds)
        tw = 0
        for w in wf:
            tw += wf[w]
        return wf, tw
    def tagLineAnalysis(self):
        wds = []
        for line in self.taglines:
            for word in line.split(" "):
                wds.append(word)
        for i in range(len(wds)):
            wds[i]= cleanWord2(wds [i]).lower()
        #print(wds)
        wf = Counter (wds)
        tw = 0
        for w in wf:
            tw += wf[w]
        tlc =len(self.taglines)
        return wf, tw, tlc
    def wholeJokeAnalysis(self):
        jtw = 0
        jwds = []
        swds = self.setup.split(" ")
        for i in range(len(swds)):
            jwds.append(cleanWord2(swds [i]).lower())
        pwds = self.punchline.split(" ")
        for i in range(len(pwds)):
            jwds.append(cleanWord2(pwds [i]).lower())

        twds = []
        for line in self.taglines:
            for word in line.split(" "):
                twds.append(word)
        for i in range(len(twds)):
            jwds.append(cleanWord2(twds [i]).lower())
        jwf = Counter (jwds)
        for w in jwf:
            jtw += jwf[w]
        return jwf, jtw

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

class Model():
    def __init__(self, jokes):
        self.lexicon = {}
        self.jokes = jokes
        self.setups = []
        self.setupAllWords = 0
        self.punchlines = []
        self.punchlineAllWords = 0
        self.taglines = []
        self.tagLineAllWords = 0
        self.wholeJokes =[]
        self.wholeJokeAllWords = 0
        self.totalNumberOfTaglines = 0
        self.similarityScore = 0
        for joke in jokes:
            swf, sTotalWords = joke.setupAnalysis()
            self.setupAllWords += sTotalWords
            self.setups.append([swf,sTotalWords])

            pwf, pTtotalWords = joke.punchLineAnalysis()
            self.punchlineAllWords += pTtotalWords
            self.punchlines.append([pwf,pTtotalWords])

            twf, tTotalWords, numberOfTagLines = joke.tagLineAnalysis()
            #print(tTotalWords)
            if(tTotalWords > 0):
                self.tagLineAllWords += tTotalWords
                self.taglines.append([twf,tTotalWords])
                self.totalNumberOfTaglines += numberOfTagLines

            jtf, jTotalWords = joke.wholeJokeAnalysis()
            for word in jtf:
                existing = self.lexicon.get(word,0)
                existing += jtf.get(word)
                self.lexicon.update({word:existing})
            self.wholeJokeAllWords += jTotalWords
            self.wholeJokes.append([jtf,jTotalWords])
        for joke in jokes:
            goDeeper = True
            i = 1
            similarityScore = 0
            while(goDeeper): 
                mp = []
                jp = []
                for j in self.jokes:
                    for p in j.getAllPhrases(i):
                        mp.append(p)
                modelPhrases = Counter(mp)
                common = self.mostCommonNWords(modelPhrases,15)
                if (common[0][1]==1):
                    goDeeper = False
                    break
                for phrase in joke.getAllPhrases(i):
                    jp.append(phrase)
                for commonWord in common:
                    if commonWord[0] in jp:
                        similarityScore += (2**i)
                i +=1
                self.similarityScore += similarityScore/jTotalWords
        self.similarityScore = self.similarityScore/len(jokes)
    def mostCommonNWords(self,check,n):
        wfs = sorted (check.items(), key = operator.itemgetter(1), reverse=True)
        ml = min(len(wfs),n)
        #Reverse the list because barh plots items from the bottom
        return (wfs [ 0:ml ] [::])
            
    def compareJokeToModel(self,joke):
        numberOfJokes = len(self.jokes)
        swf, sTotalWords = joke.setupAnalysis()
        pwf, pTtotalWords = joke.punchLineAnalysis()
        twf, tTotalWords, numberOfTagLines = joke.tagLineAnalysis()
        jtf, jTotalWords = joke.wholeJokeAnalysis()
        percentErrorSetup = (abs(sTotalWords - (self.setupAllWords/numberOfJokes))/(self.setupAllWords/numberOfJokes))*100
        percentErrorPunchline = (abs(pTtotalWords - (self.punchlineAllWords/numberOfJokes))/(self.punchlineAllWords/numberOfJokes))*100
        percentErrorTagline = (abs(tTotalWords - (self.tagLineAllWords/numberOfJokes))/(self.tagLineAllWords/numberOfJokes))*100
        percentErrorTagLineNumber = (abs(numberOfTagLines - (self.tagLineAllWords/self.totalNumberOfTaglines))/(self.tagLineAllWords/self.totalNumberOfTaglines))*100
        percentErrorTotal = (abs(jTotalWords - (self.wholeJokeAllWords/numberOfJokes))/(self.wholeJokeAllWords/numberOfJokes))*100


        '''
        Look at common words and phrases, assigning higher probability the more over lap there is with more niche phrases
        '''
        goDeeper = True
        i = 1
        similarityScore = 0
        while(goDeeper): 
            mp = []
            jp = []
            for j in self.jokes:
                for p in j.getAllPhrases(i):
                    mp.append(p)
            modelPhrases = Counter(mp)
            common = self.mostCommonNWords(modelPhrases,15)
            if (common[0][1]==1):
                goDeeper = False
                break
            for phrase in joke.getAllPhrases(i):
                jp.append(phrase)
            for commonWord in common:
                if commonWord[0] in jp:
                    similarityScore += (2**i)
            i +=1
        similarityScore = similarityScore/jTotalWords
        return percentErrorSetup, percentErrorPunchline,percentErrorTagline,percentErrorTagLineNumber,percentErrorTotal,similarityScore
        '''
        print(sTotalWords)
        print(pTtotalWords)
        print(tTotalWords)
        print(numberOfTagLines)
        print(jTotalWords)
        print("---------")
        print(percentErrorSetup)
        print(percentErrorPunchline)
        print(percentErrorTagline)
        print(percentErrorTagLineNumber)
        print(percentErrorTotal)
        '''

    def printStatistics(self):
        numberOfJokes = len(self.jokes)
        print("Setups Average Length:",str(self.setupAllWords/numberOfJokes))
        print("Punchlines Average Length:",str(self.punchlineAllWords/numberOfJokes))
        print("Taglines Average Length:",str(self.tagLineAllWords/self.totalNumberOfTaglines))
        print("Taglines per Joke on Average:",str(self.totalNumberOfTaglines/numberOfJokes))
        print("Average Similarity Score :",str(self.similarityScore))
        print("All Jokes Average Length:",str(self.wholeJokeAllWords/numberOfJokes))

def loadJokes(comedian):
    allJokes = []
    jokesInRoutines = [f for f in os.listdir(comedian+'/') if (os.path.isfile(os.path.join(comedian+'/', f)) and f[-5:]=="jokes")]
    #print(jokesInRoutines)
    for jokes in jokesInRoutines:
        with open(comedian+"/"+jokes,"r") as lines:
            setup = ""
            punchline = ""
            inSetup = False
            inJoke = False
            inTaglines = False
            taglines = []
            for line in lines:
                #print(line)
                if (line == "----Setup----\n"):
                    inSetup = True
                    inJoke = False
                    inTaglines = False
                    if(not(setup == "")):
                        joke = Joke(setup,punchline)
                        for tagline in taglines:
                            joke.addTagLine(tagline)
                        allJokes.append(joke)
                        setup = ""
                        punchline = ""
                        taglines = []
                elif (line == "----Punchline----\n"):
                    inSetup = False
                    inJoke = True
                    inTaglines = False
                elif (line == "----Tagline(s)----\n"):
                    inSetup = False
                    inJoke = False
                    inTaglines = True
                elif(inSetup):
                    setup += line
                elif(inJoke):
                    punchline += line
                elif(inTaglines):
                    taglines.append(line)
            if(not(setup == "")):
                        joke = Joke(setup,punchline)
                        for tagline in taglines:
                            joke.addTagLine(tagline)
                        allJokes.append(joke)
                        setup = ""
                        punchline = ""
                        taglines = []
    return allJokes             

def findJokes(comedian):
    '''
    A joke with no setup is most likely a tagline beloning to the most recent joke (Naive Approach)
    Alternativly, look all jokes setup's and look for correlations, but give additional weight to the most recent joke 
    '''
    transcripts = [f for f in os.listdir(comedian+'/') if (os.path.isfile(os.path.join(comedian+'/', f)) and f[-3:]=="txt")]
    for transcript in transcripts:
        jokes = []
        with open(comedian+"/"+transcript,"r") as lines:
            setup = ""
            joke = ""
            mostRecentJoke = ""
            for line in lines:
                endOfJoke = False 
                for indicator in jokeIndicators:
                    if(indicator in line):
                        endOfJoke = True
                if(not endOfJoke):
                    #print(line)
                    setup += joke
                    joke = line
                #
                elif ():
                    setup = ""
                    joke = ""    
                else:
                    if (setup != ""):
                        mostRecentJoke = Joke(setup,joke)
                        jokes.append(mostRecentJoke)
                    else:
                        mostRecentJoke.addTagLine(joke)
                    joke = ""
                    setup = ""
            mostRecentJoke = Joke(setup,joke)
            jokes.append(mostRecentJoke)
            #print("---End of Transcript---")
        with open(comedian+"/"+transcript[0:-3]+"jokes","w") as parsedJokes:
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
        if((char == '\n') or (char ==']') or (previousChar == "…" and char == " " and nextChar =="[") or (char == "♪" and nextChar !=" ") or (char == "." and nextChar != '”') or (char == '”' and previousChar in [".","?","!"])):
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

def createTrainingData(jokes, percentage):
    #Given a set of known jokes, randomly remove a given percentage of the dataset.
    #Use the remaining jokes to train the model, then see if it is properly able to clasify the removed jokes
    trainingData = []
    testData = []
    for joke in jokes:
        roll = random.randint(1,100)
        if ((roll <= percentage) and len(testData) < ((percentage/100) * len(jokes))):
            testData.append(joke)
        else:
            trainingData.append(joke)
    return trainingData,testData


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
            #print(transcript)
            #Based off of how scrapes from the loft organizes its transripts, this will be the name of the routine
            resultName = transcript.split("comedy/")[1].replace("\n","").replace("/","")
            #crawling raw dataset
            scraped = requests.get(transcript[:-1], headers={'User-agent': 'Mozilla/5.0'}).text
            #print(scraped)
            raw_result = cleanHtml(scraped)
            #print(raw_result)
            cleaned_result = cleanData(raw_result)
            #save result into the txt file
            with open(path+"/"+resultName+".txt", "w") as result_clean_file:
            	result_clean_file.write(cleaned_result)
            #print(inBrackets(cleaned_result))
#main()
#findJokes("Dave_Chappelle")

def fulltrainingAgaisntJokeInTrainingSet(comedian):
    results = open("FullTraining.txt","w")
    jokes = loadJokes(comedian)
    trainingData, testData = createTrainingData(jokes,0)
    trainedModel = Model(trainingData)
    #print(trainedModel.mostCommonNWords(15))
    trainedModel.printStatistics()
    percentErrorSetupT = 0
    percentErrorPunchlineT = 0 
    percentErrorTaglineT = 0
    percentErrorTagLineNumberT = 0
    percentErrorTotalT = 0
    similarityScoreT = 0
    '''
    randomJoke = jokes[random.randint(0,len(jokes)-1)]
    print(randomJoke.toString())
    a,b,c,d,e =trainedModel.compareJokeToModel(randomJoke)
    percentErrorSetupT += a
    percentErrorPunchlineT += b
    percentErrorTaglineT += c
    percentErrorTagLineNumberT += d
    percentErrorTotalT += e
    '''
    for joke in trainingData:
        results.write(joke.toString()+"\n")
        a,b,c,d,e,f =trainedModel.compareJokeToModel(joke)
        results.write("Similarity Score:"+str(f)+"\n")
        results.write("Percent Error Setup:"+str(a)+"%\n")
        results.write("Percent Error Punchline:"+str(b)+"%\n")
        results.write("Percent Error TagLine:"+str(c)+"%\n")
        results.write("Percent TagLine Number:"+str(d)+"%\n")
        results.write("Percent Error Total:"+str(e)+"%\n")

        percentErrorSetupT += a
        percentErrorPunchlineT += b
        percentErrorTaglineT += c
        percentErrorTagLineNumberT += d
        percentErrorTotalT += e
        similarityScoreT += f
    percentErrorSetupT = percentErrorSetupT/len(trainingData)
    percentErrorPunchlineT = percentErrorPunchlineT/len(trainingData)
    percentErrorTaglineT = percentErrorTaglineT/len(trainingData)
    percentErrorTagLineNumberT = percentErrorTagLineNumberT/len(trainingData)
    percentErrorTotalT = percentErrorTotalT/len(trainingData)
    similarityScoreT = similarityScoreT/len(trainingData)

    print(percentErrorSetupT)
    print(percentErrorPunchlineT)
    print(percentErrorTaglineT)
    print(percentErrorTagLineNumberT)
    print(percentErrorTotalT)
    print(similarityScoreT)

def testAgainstNPercent(comedian,n):
    results = open("AgainstSelfTraining.txt","w")
    jokes = loadJokes(comedian)
    trainingData, testData = createTrainingData(jokes,n)
    trainedModel = Model(trainingData)
    #print(trainedModel.mostCommonNWords(15))
    trainedModel.printStatistics()
    percentErrorSetupT = 0
    percentErrorPunchlineT = 0 
    percentErrorTaglineT = 0
    percentErrorTagLineNumberT = 0
    percentErrorTotalT = 0
    similarityScoreT = 0
    '''
    randomJoke = jokes[random.randint(0,len(jokes)-1)]
    print(randomJoke.toString())
    a,b,c,d,e =trainedModel.compareJokeToModel(randomJoke)
    percentErrorSetupT += a
    percentErrorPunchlineT += b
    percentErrorTaglineT += c
    percentErrorTagLineNumberT += d
    percentErrorTotalT += e
    '''
    success = 0
    fail = 0
    for joke in testData:
        results.write(joke.toString()+"\n")
        a,b,c,d,e,f =trainedModel.compareJokeToModel(joke)
        results.write("Similarity Score:"+str(f)+"\n")
        results.write("Percent Error Setup:"+str(a)+"%\n")
        results.write("Percent Error Punchline:"+str(b)+"%\n")
        results.write("Percent Error TagLine:"+str(c)+"%\n")
        results.write("Percent TagLine Number:"+str(d)+"%\n")
        results.write("Percent Error Total:"+str(e)+"%\n")

        if(f >= trainedModel.similarityScore):
            success += 1
        else:
            fail += 1

        percentErrorSetupT += a
        percentErrorPunchlineT += b
        percentErrorTaglineT += c
        percentErrorTagLineNumberT += d
        percentErrorTotalT += e
        similarityScoreT += f
    print("Success Rate:"+str((success/len(testData)*100))+"%")
    print("Error Rate:"+str((fail/len(testData)*100))+"%")
    percentErrorSetupT = percentErrorSetupT/len(trainingData)
    percentErrorPunchlineT = percentErrorPunchlineT/len(trainingData)
    percentErrorTaglineT = percentErrorTaglineT/len(trainingData)
    percentErrorTagLineNumberT = percentErrorTagLineNumberT/len(trainingData)
    percentErrorTotalT = percentErrorTotalT/len(trainingData)
    similarityScoreT = similarityScoreT/len(trainingData)

    print(percentErrorSetupT)
    print(percentErrorPunchlineT)
    print(percentErrorTaglineT)
    print(percentErrorTagLineNumberT)
    print(percentErrorTotalT)
    print(similarityScoreT)
        

    
#fulltrainingAgaisntJokeInTrainingSet("Dave_Chappelle")
testAgainstNPercent("Dave_Chappelle",20)
####################
#we can use the wf part to count the number of word frequencies
####################
