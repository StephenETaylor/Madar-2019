#!/usr/bin/env python3

"""
    Read through the TRAIN.tsv file, and write the tweets into different 
    directories depending out the language.
    The file is a tab separated CSV file with 6 columns
    column 4 is the country of the tweeter, and column 5 is the tweet
    We'll write them to a directory, out-21c or out-21w depending on whether the
    -c  (character split) flag is given on the command line.
"""

import math
import sys

Charmode = False
Filename = 'MADAR-Shared-Task-Subtask-2/TRAIN-features.tsv'
OutputPrefix = 'out'
Extension = '.txt'

def command_line():
    global Charmode, Filename , OutputPrefix , Extension 

    state = 0
    for w in sys.argv:
        if state == 0:
            state = 1
        #elif state == 2:
#
        elif w == '-c': Charmode = True
        elif w == '-w': Charmode = False
        else: # must be input file name
            Filename = w

def main():
    if Charmode:
        print('Char mode')
        OutputDirectory = OutputPrefix+'-21c/'
    else:
        print('Word mode')
        OutputDirectory = OutputPrefix+'-21w/'
    print('Input', Filename)
    print('OutputPrefix', OutputDirectory)

    outputfiles = dict()
    with open(Filename) as fi:
        for lin in fi:
            if lin == '\n' or lin[0] == '#' : continue
            row = lin.strip().split('\t')
            country = row[4]
            tweet = row[5]
            fo = outputfiles.get(country,None)
            if type(fo) == type(None):
                fo = open(OutputDirectory+country+Extension, 'w')
                outputfiles[country] = fo
            # preprocess tweet
            if Charmode:
                oldline = tweet
                tweet = ""
                for word in oldline.split():
                    tweet += ('<w> ')
                    for ch in word:
                        tweet += (ch)
                        tweet += (' ')
                    tweet += ('</w> ')

            # write to file
            fo.write(tweet)
            fo.write('\n')

    for k,fo in outputfiles.items():
        fo.close()




if __name__ == '__main__':
    command_line()
    main()
