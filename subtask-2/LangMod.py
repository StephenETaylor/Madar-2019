import kenlm
import math
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin

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
            sumxx = 0
            for xrc in x:
                #xx = math.exp(xrc), but there's an exp in transform
                sumxx += xrc # was xx
            x = (1/sumxx) * x
            X[r, :] = x
            
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
        
