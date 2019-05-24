Makefile  
  describes how to build the language models for the 21 countries, once the
  tweets have been downloaded.

The MADAR-Obtain-Tweets.py file is part of the data distribution for the
shared task, and is not included here.  

lmplz  is the kenlm tool for building language models

build_binary is a kenlm tool for speeding up language models.

I built lmplz and build_binary from the code at https://github.com/kpu/kenlm

out-21w, out-21c, model.21w, model.21c are empty directories which are populated
by programs described in Makefile

judge-tweet4.py produces the predictions for subtask-2, based on a 
development file or a testfile, depending on the value of
DEVTEST set in line 38.  Currently it is set to False, meaning that the 
Test input and output file names should be used; you can see from the code that
two files are used for the test, a Features.tsv file, which I built without
using the Makefile, but using a MADAR*TEST*features.tsv file and
MADAR-Obtain-tweets.py, and MADAR-Twitter-Subtask-2.TEST.user-label.tsv,
which in spite of its name contains no country labels.  

The output for the test was named ZCU-NLP1.subtask2.test, and it contains
both the user names and their predicted country labels.
