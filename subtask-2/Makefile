ST0	=MADAR-Twitter-Subtask-2
ST2	=MADAR-Shared-Task-Subtask-2/
DISCOUNT = --discount_fallback
SWITCHES = -o 5 ${DISCOUNT}
countries = Algeria Kuwait Palestine Tunisia Bahrain Lebanon Qatar United_Arab_Emirates Djibouti Libya Saudi_Arabia Yemen Egypt Mauritania Somalia Iraq Morocco Sudan Jordan Oman Syria
csuff 	= $(addsuffix .binary,${countries})
charmodels=$(addprefix model.21c/,${csuff})
wordmodels=$(addprefix model.21w/,${csuff})

default:	${charmodels} ${wordmodels}

# make up the out-21w out-21c directories
${ST2}outs:	${ST2}TRAIN-features.tsv
	python3 spill-tweets.py -w $^
	python3 spill-tweets.py -c $^

# download the training and dev tweets
${ST2}TRAIN-features.tsv:	${ST0}.TRAIN.user-tweets-features.tsv 
	${ST2}MADAR-Obtain-Tweets.py $^ $@

${ST2}DEV-features.tsv:	${ST0}.DEV.user-tweets-features.tsv 
	${ST2}MADAR-Obtain-Tweets.py $^ $@

# make the kenlm language models
model.21w/%.binary:	out-21w/%.txt
	lmplz ${SWITCHES} <$^ >$(basename $@).arpa
	build_binary $(basename $@).arpa $@

model.21c/%.binary:	out-21c/%.txt
	lmplz ${SWITCHES} <$^ >$(basename $@).arpa
	build_binary $(basename $@).arpa $@

%:	model.21c/%.binary

%:	model.21w/%.binary


