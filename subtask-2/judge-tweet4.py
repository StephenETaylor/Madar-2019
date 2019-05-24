#!/usr/bin/env python3
"""
    judge-tweets4:
    reads test file by default, and writes user-guesses, which 
    is the result of a hard vote on each of the users tweets.   As of 10.5.2019
    313/500 of the votes on the test file result in Saudi_Arabia, although
    the tweets are 49+% accurately assigned.

    judge-tweets1-3:
    this program reads the TRAIN-features.tsv file,
    and by default, reads the DEV-features.tsv file.
    Inspired by semi-aimless experiments in F3b.py, it attempts to 
    combine various information sources to obtain dialect results on the 
    tweets task.

    Format of file, according to ../README.txt:
    col 0: twitter name
    col 1: tweet ID
    col 2: twitter language guess, always 'ar'
    col 3: comma separated guesses for the 26 cities
    col 4: answer to ignore
    col 5: downloaded tweet
"""
from LangMod import LangModels
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
import sys
import time

DEVTEST = False   #True for Development testing, False for final test set

TrainFile = 'MADAR-Shared-Task-Subtask-2/TRAIN-features.tsv'

FinalTestFile = 'MADAR-Shared-Task-Subtask-2-Test/Features.tsv'
DevFile = 'MADAR-Shared-Task-Subtask-2/DEV-features.tsv'

UserDevFile = 'MADAR-Shared-Task-Subtask-2/MADAR-Twitter-Subtask-2.DEV.user-label.tsv'
UserTestFile = 'MADAR-Shared-Task-Subtask-2-Test/MADAR-Twitter-Subtask-2.TEST.user-label.tsv'

OutputTest = 'ZCU-NLP1.subtask2.test'
OutputDev = 'ZCU-NLP1.subtask2.DEV'


if DEVTEST:
    UserFile = UserDevFile
    OutputFile = OutputDev
    TestFile = DevFile
    InterFile = 'tweets-guess4'
else:
    UserFile = UserTestFile
    OutputFile = OutputTest
    TestFile = FinalTestFile
    InterFile = 'sweets-guess4'



X1_train = None
X2_train = None
y_train = None
X1_test = None
X2_test = None
y_test = None
States = None

def command_line():
    global TrainFile, TestFile, OutputFile
    if len(sys.argv) > 1 and sys.argv[1] != '-': 
        InputFile = sys.argv[1] 
    if len(sys.argv) > 2 and sys.argv[2] != '-': 
        InputFile = sys.argv[2] 
    if len(sys.argv) > 3 and sys.argv[3] != '-':
        OutputFile = sys.argv[3] 

def readFeaturesTsv(InputFile):
    
    """
    reads an Input file in the task-2 format
    returns arrays:
        X0      userid
        X1      'features' from the file, the 26 city probabilities
        X2      tweets from the file
        y       country for each item

    """
    X0 = [] # userid
    X1 = [] # features
    X2 = [] # tweets from features file
    y = []  # country, if given, from features file
    with open(InputFile) as fi:
        testOrDev = None
        for lin in fi:
            if len(lin) == 0 or lin == '\n' or lin[0] == '#': continue
            line = lin.strip().split('\t')
            X0.append(line[0])
            if line[3] == '<NIL>':
                pfeats = [(1/26)]*26 # no information from this...
            else:
                feats = line[3].split(',')
                pfeats = [0]*(len(feats))
                if len(pfeats) != 26: cry()
                sumFeats = 0
                for i,f in enumerate(feats):
                    g = float(f)
                    pfeats[i] = g
                    sumFeats += g

                if sumFeats == 1: 
                    pass
                else:
                    for i,p in enumerate(pfeats):
                        pfeats[i] = p/sumFeats      # normalize to sum to 1
            X1.append(pfeats)
            if len(line) == 5: # if this is a test or dev file with state
                if testOrDev == 'test' or testOrDev == None:
                    X2.append(line[4])
                    y.append(line[4])       # in test this will be useless
                    testOrDev = 'test'
                else:
                    sys.stderr.write('messed up file, mixed 5 and 6 field lines\n')
                    sys.exit(1)
            else: # line len should be 6, field 5, (last) is tweet
                if testOrDev == 'Dev' or testOrDev == None:
                    X2.append(line[5])
                    y.append(line[4])       # in test this will be useless
                    testOrDev = 'Dev'
                else:
                    sys.stderr.write('messed up file, mixed 6 and 5 field lines\n')
                    sys.exit(1)

    return np.array(X0),np.array(X1), X2, np.array(y)

# doll up time-keeping slightly
intstart = time.time()
def mark():
    global intstart
    temp = time.time()
    answer = 'seconds = '+str(temp-intstart)
    intstart = temp
    return answer

def main():
    global X1_train, X1_test, X2_train, X2_test, y_train, y_test
    global states

    lm_21c = LangModels('21c')
    lm_21w = LangModels('21w')
    lm_26c = LangModels('26c')
    city2state = {'RAB':'Morocco','FES':'Morocco',
            'ALG':'Algeria', 'SFX':'Tunisia', 'TUN':'Tunisia',
            'TRI':'Libya', 'BEN':'Libya', 'ALX':'Egypt',
            'ASW':'Egypt', 'CAI':'Egypt', 'KHA':'Sudan', 'JER':'Jordan',
            'AMM':'Jordan', 'SAL':'Jordan', 'BEI':'Lebanon', 'DAM':'Syria',
            'ALE':'Syria', 'MOS':'Iraq', 'BAG':'Iraq',
            'BAS':'Iraq', 'DOH':'Qatar', 'MUS':'Oman',
            'MSA':'Fusha', # not a city, but one of the 26 classes
            'RIY':'Saudi_Arabia', 'JED':'Saudi_Arabia', 'SAN':'Yemen'}
    
    states = lm_21c.classes_()  # all of the states we're interested in

    print('built language models', mark())
    X0_train, X1_train, X2_train, y_train = readFeaturesTsv(TrainFile)
    X0_test, X1_test, X2_test, y_test = readFeaturesTsv(TestFile)
    print('read TRAIN and DEV', mark())

    # finished reading input file.  Process
    X3 = lm_21c.predict_proba(X2_test)
    X3_train = lm_21c.predict_proba(X2_train)

    userdict = dict()


    # do a little voting run-through
    cl3 = lm_21c.classes_()

    if False:
        X4 = lm_21w.predict_proba(X2_test)
        X4_train = lm_21w.predict_proba(X2_train)

        c_345gram_vec = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5))
        p_c_345gram_vec = Pipeline([
                ('char-345grams', c_345gram_vec)
                ,
                ('mnb', MultinomialNB())
                    ])   #.fit(X2_train, y_train)

        w_1_2gram_vec = TfidfVectorizer(analyzer='word',ngram_range=(1,2))
        p_w_1_2gram_vec = Pipeline([
                ('word-unigrams', w_1_2gram_vec),
                ('mnb', MultinomialNB())
                    ])    .fit(X2_train, y_train)

        mnb1 = MultinomialNB()
        y2_test = experiment1(mnb1,'mnb1')

        print(time.time())
        total = X1_test.shape[0]
        right1 = right2 = right3 = right4 =rightv= 0

    # try voting with a little more attention?

        XX = np.concatenate((p_w_1_2gram_vec.predict_proba(X2_train),
                                      mnb1.predict_proba(X1_train),
                                      X3_train,
                                      X4_train), axis = 1);
        XT = np.concatenate((p_w_1_2gram_vec.predict_proba(X2_test),
                                      mnb1.predict_proba(X1_test),
                                      X3,
                                      X4), axis = 1);
        print(time.time())

    import pickle
    with open('X3','wb') as fi:
        pickle.dump((X3, X3_train,y_train),fi)

    print('ready',mark())

    y1_test = KNeighborsClassifier(n_neighbors=31).fit(X3_train,y_train).predict( X3)
    #print('knn-21c',accuracy(y_test,y1_test), mark())

    print('tweets-classified', mark())

    with open(InterFile,'w') as fo:
        for y1,user in zip(y1_test, X0_test): #write out intermediate file
            #fo.write(user)
            #fo.write('\t')
            fo.write(y1)
            fo.write('\n')
            uud = userdict.get(user,0)
            if uud == 0:
                uud = dict()
                userdict[user] = uud
            cnt = uud.get(y1,0)
            uud[y1] = cnt+1

    print('wrote intermediate file', mark())

    with open(OutputFile,'w') as fo:
        with open(UserFile) as fi:
            for lin in fi:
                if len(lin) == 0 or lin == '\n' or lin[0] == '#' : continue
                line = lin.strip().split()
                if len(line) > 1:
                    # must if dev
                    gold = line[1]
                user = line[0]
                uud = userdict[user]
                vote = []
                for k,v in uud.items():
                    vote.append((v,k))
                vote.sort()
                fo.write(user)
                fo.write('\t')
                fo.write(vote[-1][1])
                fo.write('\n')

    print('done', mark())

def experiment1(item,tag):
    return experiment(item,tag,X1_train,y_train,X1_test,y_test)

def experiment2(item,tag):
    return experiment(item,tag,X2_train,y_train,X2_test,y_test)

def experiment(item,tag,X_train,y_train,X_test,y_test):
    start = time.time()
    item.fit(X_train, y_train)
    y_predicted = item.predict(X_test)
    interval = time.time()-start
    print(tag,accuracy(y_test, y_predicted),'seconds=',interval)
    return y_predicted

def accuracy(y_test,y_predicted):
    correct = 0
    total = 0
    for gold,pred in zip(y_test,y_predicted):
        total += 1
        if gold == pred: correct += 1
    return correct/total

def paccuracy(a,b):
    print('accuracy =', accuracy(a,b))





if  __name__ == '__main__':
    command_line()
    main()

