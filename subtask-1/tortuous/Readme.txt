Makefile contains procedures for building two text directories:
out.sw and out.sc, but you actually want to build FOUR text directories:
out.6w out.6c   and out.26w out.26c

In order to do this, you'll 
1) edit split-tsv.py to look at the appropriate TRAIN file, either -6- or -26-
2) make out.sc 
3) make out.sw
4) rename out.sc to the appropriate name, either out.6c or out.26c
5) similarly for out.sw
re-edit split-tsv.py, repeat.

We'd edit the Makefile, but we don't expect to test it, so you're better off
deciding yourself whether you want to be that patient.

Similarly the makefile builds five model directories: 
model.6c model.6w model.26c model.26w

We actually built the model directory first, then renamed it to model.26w
refilled the text directory, then called outs, ... renamed, etc.

The model26 goals have not been tested, but are included to document what
seems like it should work.  The directories themselves are not created by
the relevant programs, and must be added by hand.

However the model.6c and model.6w directories *were* filled using Makefile,
and a similar, simpler makefile was used for subtask-2

the *score* goal in the Makefile uses the scoring file distributed by 
the task organizers.  The *dev.gold file can be built from the the MADAR*DEV*tsv
file using the unix shell command 'cut -f 2'.  

Predictions were made using test26SET1.py, which has the name of
the training file built in and takes a test file as a parameter.  
If no parameter is given, it defaults to the development file.
It writes to a filename based on the input file with added .clf extension.




