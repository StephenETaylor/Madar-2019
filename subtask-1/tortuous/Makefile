# describes setup
# make outs; make lm
# tried building without --discount_fallback, to see if we get different models
# apparently not; but model build can fail if flag not present.
DISCOUNT = --discount_fallback
SWITCHES = -o 5 ${DISCOUNT}
DIALECTS = ALE ALG ALX AMM ASW BAG BAS BEI BEN CAI DAM DOH FES JED JER KHA MOS MSA MUS RAB RIY SAL SAN SFX TRI TUN 
models=model/ALE.binary model/ALG.binary model/ALX.binary model/AMM.binary model/ASW.binary model/BAG.binary model/BAS.binary model/BEI.binary model/BEN.binary model/CAI.binary model/DAM.binary model/DOH.binary model/FES.binary model/JED.binary model/JER.binary model/KHA.binary model/MOS.binary model/MSA.binary model/MUS.binary model/RAB.binary model/RIY.binary model/SAL.binary model/SAN.binary model/SFX.binary model/TRI.binary model/TUN.binary 

lm:	${models}

score:	
	python3 MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-DID-Scorer-v.0.1.py MADAR-SHARED-TASK-third-release-8Mar2019/MADAR-Shared-Task-Subtask-1/MADAR-Corpus-26-dev.gold pyrite

out.sw:	split-tsv.py
	mkdir -p $@
	rm -f outs
	python3 $^ -w -oprefix out.sw/MADAR_TRAINING_SAMPLE
	ln -s out.sw outs

out.sc:	split-tsv.py
	mkdir -p $@
	rm -f outs
	python3 $^ -c -oprefix out.sc/MADAR_TRAINING_SAMPLE
	ln -s out.sc outs

model/ALE.binary:	outs/MADAR_TRAINING_SAMPLE-ALE.txt
	lmplz ${SWITCHES} <$^ >model/ALE.arpa
	build_binary model/ALE.arpa model/ALE.binary 

model/ALG.binary:	outs/MADAR_TRAINING_SAMPLE-ALG.txt
	lmplz ${SWITCHES} <$^ >model/ALG.arpa
	build_binary model/ALG.arpa model/ALG.binary

model/ALX.binary:	outs/MADAR_TRAINING_SAMPLE-ALX.txt
	lmplz ${SWITCHES} <$^ >model/ALX.arpa
	build_binary model/ALX.arpa model/ALX.binary

model/AMM.binary:	outs/MADAR_TRAINING_SAMPLE-AMM.txt
	lmplz ${SWITCHES} <$^ >model/AMM.arpa
	build_binary model/AMM.arpa model/AMM.binary

model/ASW.binary:	outs/MADAR_TRAINING_SAMPLE-ASW.txt
	lmplz ${SWITCHES} <$^ >model/ASW.arpa
	build_binary model/ASW.arpa model/ASW.binary

model/BAG.binary:	outs/MADAR_TRAINING_SAMPLE-BAG.txt
	lmplz ${SWITCHES} <$^ >model/BAG.arpa
	build_binary model/BAG.arpa model/BAG.binary

model/BAS.binary:	outs/MADAR_TRAINING_SAMPLE-BAS.txt
	lmplz ${SWITCHES} <$^ >model/BAS.arpa
	build_binary model/BAS.arpa model/BAS.binary

model/BEI.binary:	outs/MADAR_TRAINING_SAMPLE-BEI.txt
	lmplz ${SWITCHES} <$^ >model/BEI.arpa
	build_binary model/BEI.arpa model/BEI.binary

model/BEN.binary:	outs/MADAR_TRAINING_SAMPLE-BEN.txt
	lmplz ${SWITCHES} <$^ >model/BEN.arpa
	build_binary model/BEN.arpa model/BEN.binary

model/CAI.binary:	outs/MADAR_TRAINING_SAMPLE-CAI.txt
	lmplz ${SWITCHES} <$^ >model/CAI.arpa
	build_binary model/CAI.arpa model/CAI.binary

model/DAM.binary:	outs/MADAR_TRAINING_SAMPLE-DAM.txt
	lmplz ${SWITCHES} <$^ >model/DAM.arpa
	build_binary model/DAM.arpa model/DAM.binary

model/DOH.binary:	outs/MADAR_TRAINING_SAMPLE-DOH.txt
	lmplz ${SWITCHES} <$^ >model/DOH.arpa
	build_binary model/DOH.arpa model/DOH.binary

model/FES.binary:	outs/MADAR_TRAINING_SAMPLE-FES.txt
	lmplz ${SWITCHES} <$^ >model/FES.arpa
	build_binary model/FES.arpa model/FES.binary 

model/JED.binary:	outs/MADAR_TRAINING_SAMPLE-JED.txt
	lmplz ${SWITCHES} <$^ >model/JED.arpa
	build_binary model/JED.arpa model/JED.binary

model/JER.binary:	outs/MADAR_TRAINING_SAMPLE-JER.txt
	lmplz ${SWITCHES} <$^ >model/JER.arpa
	build_binary model/JER.arpa model/JER.binary

model/KHA.binary:	outs/MADAR_TRAINING_SAMPLE-KHA.txt
	lmplz ${SWITCHES} <$^ >model/KHA.arpa
	build_binary model/KHA.arpa model/KHA.binary

model/MOS.binary:	outs/MADAR_TRAINING_SAMPLE-MOS.txt
	lmplz ${SWITCHES} <$^ >model/MOS.arpa
	build_binary model/MOS.arpa model/MOS.binary

model/MSA.binary:	outs/MADAR_TRAINING_SAMPLE-MSA.txt
	lmplz ${SWITCHES} <$^ >model/MSA.arpa
	build_binary model/MSA.arpa model/MSA.binary

model/MUS.binary:	outs/MADAR_TRAINING_SAMPLE-MUS.txt
	lmplz ${SWITCHES} <$^ >model/MUS.arpa
	build_binary model/MUS.arpa model/MUS.binary

model/RAB.binary:	outs/MADAR_TRAINING_SAMPLE-RAB.txt
	lmplz ${SWITCHES} <$^ >model/RAB.arpa
	build_binary model/RAB.arpa model/RAB.binary

model/RIY.binary:	outs/MADAR_TRAINING_SAMPLE-RIY.txt
	lmplz ${SWITCHES} <$^ >model/RIY.arpa
	build_binary model/RIY.arpa model/RIY.binary

model/SAL.binary:	outs/MADAR_TRAINING_SAMPLE-SAL.txt
	lmplz ${SWITCHES} <$^ >model/SAL.arpa
	build_binary model/SAL.arpa model/SAL.binary

model/SAN.binary:	outs/MADAR_TRAINING_SAMPLE-SAN.txt
	lmplz ${SWITCHES} <$^ >model/SAN.arpa
	build_binary model/SAN.arpa model/SAN.binary

model/SFX.binary:	outs/MADAR_TRAINING_SAMPLE-SFX.txt
	lmplz ${SWITCHES} <$^ >model/SFX.arpa
	build_binary model/SFX.arpa model/SFX.binary

model/TRI.binary:	outs/MADAR_TRAINING_SAMPLE-TRI.txt
	lmplz ${SWITCHES} <$^ >model/TRI.arpa
	build_binary model/TRI.arpa model/TRI.binary

model/TUN.binary:	outs/MADAR_TRAINING_SAMPLE-TUN.txt
	lmplz ${SWITCHES} <$^ >model/TUN.arpa
	build_binary model/TUN.arpa model/TUN.binary


models6:	models.6c/BEI.binary models.6c/CAI.binary models.6c/DOH.binary	models.6c/MSA.binary	models.6c/RAB.binary	models.6c/TUN.binary	models.6w/BEI.binary models.6w/CAI.binary models.6w/DOH.binary	models.6w/MSA.binary	models.6w/RAB.binary	models.6w/TUN.binary	


models.6c/BEI.binary:	outs.6c/MADAR_TRAINING_SAMPLE-BEI.txt
	lmplz ${SWITCHES} <$^ >models.6c/BEI.arpa
	build_binary models.6c/BEI.arpa $@

models.6c/CAI.binary:	outs.6c/MADAR_TRAINING_SAMPLE-CAI.txt
	lmplz ${SWITCHES} <$^ >models.6c/CAI.arpa
	build_binary models.6c/CAI.arpa $@

models.6c/DOH.binary:	outs.6c/MADAR_TRAINING_SAMPLE-DOH.txt
	lmplz ${SWITCHES} <$^ >models.6c/DOH.arpa
	build_binary models.6c/DOH.arpa $@

models.6c/MSA.binary:	outs.6c/MADAR_TRAINING_SAMPLE-MSA.txt
	lmplz ${SWITCHES} <$^ >models.6c/MSA.arpa
	build_binary models.6c/MSA.arpa $@

models.6c/RAB.binary:	outs.6c/MADAR_TRAINING_SAMPLE-RAB.txt
	lmplz ${SWITCHES} <$^ >models.6c/RAB.arpa
	build_binary models.6c/RAB.arpa $@

models.6c/TUN.binary:	outs.6c/MADAR_TRAINING_SAMPLE-TUN.txt
	lmplz ${SWITCHES} <$^ >models.6c/TUN.arpa
	build_binary models.6c/TUN.arpa $@


models.6w/BEI.binary:	outs.6w/MADAR_TRAINING_SAMPLE-BEI.txt
	lmplz ${SWITCHES} <$^ >models.6w/BEI.arpa
	build_binary models.6w/BEI.arpa $@

models.6w/CAI.binary:	outs.6w/MADAR_TRAINING_SAMPLE-CAI.txt
	lmplz ${SWITCHES} <$^ >models.6w/CAI.arpa
	build_binary models.6w/CAI.arpa $@

models.6w/DOH.binary:	outs.6w/MADAR_TRAINING_SAMPLE-DOH.txt
	lmplz ${SWITCHES} <$^ >models.6w/DOH.arpa
	build_binary models.6w/DOH.arpa $@

models.6w/MSA.binary:	outs.6w/MADAR_TRAINING_SAMPLE-MSA.txt
	lmplz ${SWITCHES} <$^ >models.6w/MSA.arpa
	build_binary models.6w/MSA.arpa $@

models.6w/RAB.binary:	outs.6w/MADAR_TRAINING_SAMPLE-RAB.txt
	lmplz ${SWITCHES} <$^ >models.6w/RAB.arpa
	build_binary models.6w/RAB.arpa $@

models.6w/TUN.binary:	outs.6w/MADAR_TRAINING_SAMPLE-TUN.txt
	lmplz ${SWITCHES} <$^ >models.6w/TUN.arpa
	build_binary models.6w/TUN.arpa $@

