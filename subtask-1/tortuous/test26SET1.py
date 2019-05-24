#!/usr/bin/env python3
"""
    This program is a daughter of F3b.py, comments below.  It attempts to
    apply the 
       vc2 results reported  below 
       to an arbitrary input <pathname>, specifed on the command line,
       using the Corpus-26 Training file, (and possibly other clues)
       and it writes its classification on <pathname>.clf,
        one classification per line, as in the EXAMPLE.GOLD, EXAMPLE.PRED files.
    My plan is to omit calculations performed by F3b.py which are irrelevant
    to the vc2 computaton, in hopes of shortening the runtime, 

    results on dev set:
    wrote  5200 predictions to dev26.1.tst.clf in 387.35142374038696 seconds
    OVERALL SCORES:
    MACRO AVERAGE PRECISION SCORE: 68.68 %
    MACRO AVERAGE RECALL SCORE: 67.38 %
    MACRO AVERAGE F1 SCORE: 67.49 %
    OVERALL ACCURACY: 67.38 %


    F3b comments:
    working on word and char n-grams, using pythonier code
    This file intended to experiment with ensembles, in particular
    VotingClassifier
    
    This iteration intended to provide source code examples without 
    commented-out sections for experiments with results > 65% accuracy.

    results of run on Corpus-26 dev data, Monday 7AM 25-March-2019:
        
        26from6: G 0.551923076923077 seconds= 20.301419496536255
        26from6: H 0.6659615384615385 seconds= 65.05616497993469
        lm26c 0.6630769230769231 seconds= 1.7868480682373047
        pipeline 0.6565384615384615 seconds= 35.8366641998291
        vc 0.6786538461538462 seconds= 27.611310958862305
        vc2 0.6796153846153846 seconds= 127.31909537315369

    results of run on Corpus-6 dev data Thursday 3:30PM 26-March-2019
        26from6: G 0.9185 seconds= 12.330706119537354
        26from6: H 0.898 seconds= 17.958945274353027
        lm26c 0.883 seconds= 0.5518379211425781
        pipeline 0.9088333333333334 seconds= 37.38285684585571
        vc 0.915 seconds= 78.81152987480164
        vc2 0.9183333333333333 seconds= 197.6664171218872


"""
import kenlm
from LangMod import LangModels
import math
import numpy as np
import os
from scipy import sparse
from sklearn import svm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
import sys
import time

Trainfile26 = 'MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-26-train.tsv'
Trainfile6 = 'MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-train.tsv'

Testfile26 = 'MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-26-dev.tsv'
Testfile6 = 'MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-6-dev.tsv'

Trainfile = Trainfile26
Testfile = Testfile26

X_train = None
y_train = None
X_test = None
y_test = None


# this version of cmdline written for test26SET1.py
def cmdline():
    global Trainfile, Testfile
    Trainfile = Trainfile26
    Testfile = sys.argv[1]

# should never be called...version of cmdline used in F3b.py
def cmdlineFb3():
    global Trainfile, Testfile
    state = 0
    for s in sys.argv:
        if state == 0:
            state=1
        elif state==1 and s == '-6':
            Trainfile = Trainfile6
            Testfile = Testfile6
        elif state==1 and s == '-26':
            Trainfile = Trainfile26
            Testfile = Testfile26
        else:
            sys.stderr.write("""
usage:
    F3b {-6} {-26}
    runs a series of tests, either with the CORPUS6 or CORPUS26 train, dev files
""")


def Xy_split(fn):
    with open(fn) as fin:
        X = []
        y = []
        for lin in fin:
            spiltmilk = lin.strip().split('\t')
            ex = spiltmilk[0]
            if len(spiltmilk) == 2:
                wye = spiltmilk[1]
            else: # well, there isn't any y column, this is a test case
                wye = None
            X.append(ex)
            y.append(wye)
        return X,y

if False:
    # should I use langmod module?
    class LangModels (TransformerMixin, BaseEstimator):
        """
            This Transform class accepts a line of text, and passes it to 
            a number of kenlm language models, adding a probability estimate
            feature for each.  The language models are stored as 
            models.NNT/DIA.binary
            where  DIA is the name of a dialect, and NNT is one of 26c, 26w, 6c, 6w
            The c or w indicates preprocessing which must be done on the 
            data before passing it to the language model, and the number refers to
            which collection of models to use.
        """
        def __init__(self, modelSuffix):
            self.modelSuffix = modelSuffix
            self.lm = dict()
            for fn in os.listdir('model.'+modelSuffix):
                if fn[-7:] == '.binary':
                    dialect = fn[:-7]
                    self.lm[dialect] = (kenlm.
                                        LanguageModel('model.'+modelSuffix+'/'+fn))
            self.dialects = [x for x in self.lm.keys()]
            self.dialects.sort()
            if modelSuffix[-1] == 'c':
                self.CHARMODE = True
            else:
                self.CHARMODE = False

        def fit(self, X, y = None ):
            if type(y) == type(None): return self

            # for lm_26 with corpus 6 input, trim lists of answers

            if type(y[0]) != type('ALE'): # voting classifier switches int for class
                dinums = dict()
                for i,d in enumerate(self.dialects):
                    dinums[d] = i
                yTries = dict()
                XX = self.transform(X)

                for xx,yy in zip(XX,y):
                    di = np.argmax(xx)
                    grid = yTries.get(yy,None)
                    if grid == None:
                        grid = [0]*26
                    grid[di] += 1
                    yTries[yy] = grid
                dialects = [0]*len(yTries)
                for yy,grid in yTries.items():
                    di = np.argmax(grid)
                    dialects[yy] = self.dialects[di]
                self.dialects = dialects
                return self


            # else not numeric keys, assume the best
            dialects = dict() # create new list of dialects
            for row in y:
                # for lm_6 with corpus26 input, don't pretend to know what you don't
                if row not in self.lm: return self # don't try to expand lm
                dialects[row] = 1
            self.dialects = [k for k in dialects.keys()]
            self.dialects.sort()

            return self 

        def transform(self, X, y = None):
            out = np.ndarray(shape=(len(X),len(self.dialects)) , dtype = np.float32)
            for j,x in enumerate(X):
                text = ''
                sent = x.strip()
                if self.CHARMODE:
                    words = ['<s>']
                    for w in sent.split():
                        words .append ('<w>')
                        for ch in w:
                            words.append(ch)
                        words.append('</w>')
                    words .append('</s>')
                    swords = ' '.join(words[1:-1])

                else: # its word mode
                    words = ['<s>'] + sent.split() + ['</s>']
                    swords = sent
                lensent = len(words)
                for i,d in enumerate(self.dialects):
                    t = self.lm[d].score(swords)  # experiment with sentence score
                    #out[j,i] = t
                    out[j,i] = math.exp(t/lensent)
            # exponentiate in order to have all positive values
            # log probs can go negative
    #       out = np.exp(out)
            return out
        
        def predict_proba(self, Xtest):
            X = self.transform(Xtest)
            for r in range(X.shape[0]):
                x = X[r, :]
                for xrc in x:
                    xx = math.exp(xrc)
                
            return X

        def predict(self, Xtest):
            X = self.predict_proba(Xtest)
            i = np.argmax(X, axis = 1)
            #y = self.dialects[i]
            y = [self.dialects[j] for j in i]
            return y

        def classes_(self):
            return self.dialects

        def get_params(self,deep):
            if deep :
                alas('this branch is uncoded')
            return {'modelSuffix':self.modelSuffix}
            
def main():
    global X_train, y_train, X_test, y_test

    X_train, y_train = Xy_split(Trainfile)
    X_test, y_test = Xy_split(Testfile)

    w_unigram_vec = TfidfVectorizer(analyzer='word',ngram_range=(1,1))
    w_1_2gram_vec = TfidfVectorizer(analyzer='word',ngram_range=(1,2))
    c_123gram_vec = TfidfVectorizer(analyzer='char_wb',ngram_range=(1,3))
    c_345gram_vec = TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5))
    lm_26w = LangModels('26w')
    lm_26c = LangModels('26c')
    lm_6w = LangModels('6w')
    lm_6c = LangModels('6c')

    mnb = MultinomialNB()

    p26from6G = Pipeline([
        ('lmunion', FeatureUnion ([
             ('lm_6c', lm_6c)
             ,
             ('lm_6w', lm_6w)
             ,
             ('lm_26c', lm_26c)
              ])),
        ('mnb', MultinomialNB())
    ])

    p26from6H = Pipeline([
        ('lmunion', FeatureUnion ([
             ('lm_6c', lm_6c)
             ,
             ('lm_6w', lm_6w)
             ,
             ('lm_26c', lm_26c)
              ])),
        ('svc', svm.SVC(gamma='scale', kernel = 'poly', degree = 2))
    ])

    pipeline = Pipeline([
        ('union', FeatureUnion ([
             ('lm_26w',lm_26w),
             ('lm_26c',lm_26c),
             ('lm_6w',LangModels('6w')),
             ('lm_6c',LangModels('6c')),
            ('word-1_2grams', w_1_2gram_vec),
            ('char-345grams', TfidfVectorizer(analyzer='char',ngram_range=(3,5))),
             ('char-wb', TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5)))
                  ])),
         ('mxabs', MaxAbsScaler(copy=False)),
    #     ('mmxs', MinMaxScaler(feature_range=(1,100))), # fails for sparse arrays
    #    ('stds', StandardScaler(with_mean= False)),
         ('mnb', mnb)
    #    ('svc', svm.SVC(gamma='scale', kernel = 'poly', degree = 2))
    #     ('knn', KNeighborsClassifier(n_neighbors=15))
    ])

    p_w_1_2gram_vec = Pipeline([
            ('word-unigrams', w_1_2gram_vec),
            ('mnb', MultinomialNB())
                ]).fit(X_train, y_train)

    p_w_unigram_vec = Pipeline([
            ('word-unigrams', w_unigram_vec),
            ('mnb', MultinomialNB())
                ]).fit(X_train, y_train)

    p_w_bigram_vec = Pipeline([
            ('word-bigrams', TfidfVectorizer(analyzer='word',ngram_range=(2,2))),
            ('mnb', MultinomialNB())
                ]).fit(X_train, y_train)

    p_c_123gram_vec = Pipeline([
            ('char-123grams', c_123gram_vec)
            ,
            ('mnb', MultinomialNB())
                ]).fit(X_train, y_train)

    p_c_345gram_vec = Pipeline([
            ('char-345grams', c_345gram_vec)
            ,
            ('mnb', MultinomialNB())
                ]).fit(X_train, y_train)



    vc = VotingClassifier(voting='soft',estimators=[
         ('word-1_2grams', p_w_1_2gram_vec)
         , 
    #     ('word-bigrams', p_w_bigram_vec)
    #     , 
         ('char-345grams', p_c_345gram_vec)
         ,
         ('lm_26w',lm_26w)
         ,
         ('lm_26c', lm_26c)
         ,
         ('26from6G', p26from6G)
    ]
    #, weights = []
    ) 

    # I can't get the SVC classifier and voting-soft classifier to play together
    vc2 = VotingClassifier(voting='hard', estimators = [
        ('vc',vc),
        ('p26from6: H',p26from6H),
        ('pipeline',pipeline)]) 

    vc3 = VotingClassifier(voting='hard', estimators = [
        ('vc',vc),
        ('p26from6: G',p26from6G),
        ('pipeline',pipeline)]) 

    # test
    if False:
        # Each of these experiments, I believe, repeats portions of earlier
        # experiments using its components.  So I won't do any of them 
        # in test6SET1.py.  Possibly I could get messed up by side-effects...
        experiment(p26from6G,'26from6: G')
        experiment(p26from6H,'26from6: H')
        experiment(lm_6w, 'lm_6w')
        experiment(lm_6c, 'lm_6c')
        experiment(lm_26w,'lm26w')
        experiment(lm_26c,'lm26c')
        experiment(p_w_unigram_vec, 'p_w_unigram_vec')
        experiment(p_w_1_2gram_vec, 'p_w_1_2gram_vec')
        experiment(p_c_123gram_vec, 'p_c_123gram_vec')
        experiment(p_c_345gram_vec, 'p_c_345gram_vec')
        experiment(pipeline,'pipeline')
        experiment(vc,'vc')
        experiment(vc2,'vc2')
        experiment(vc2,'vc3')

    engine = vc2
    engineName = 'vc2'
    
    # testing code
    sys.stderr.write(sys.argv[0])
    sys.stderr.write('\n')
    sys.stderr.write(engineName)
    sys.stderr.write('\n')
    start = time.time()            #code lifted from experiment, below

    engine.fit(X_train, y_train)
    y_predicted = engine.predict(X_test)
    with open(Testfile+'.clf','w') as fi:
        for yy in y_predicted:
            fi.write(yy)
            fi.write('\n')

    interval = time.time()-start
    print ('wrote ',len(y_predicted),'predictions to',Testfile+'.clf','in',interval, 'seconds') 



def experiment(item,tag):
    start = time.time()
    item.fit(X_train, y_train)
    y_predicted = item.predict(X_test)
    interval = time.time()-start
    print(tag,accuracy(y_test, y_predicted),'seconds=',interval)

def accuracy(y_test,y_predicted):
    correct = 0
    total = 0
    for gold,pred in zip(y_test,y_predicted):
        total += 1
        if gold == pred: correct += 1
    return correct/total

def paccuracy(a,b):
    print('accuracy =', accuracy(a,b))
    #print(classification_report(y_test, y_predicted))

if __name__ == '__main__':
    cmdline()
    main()
