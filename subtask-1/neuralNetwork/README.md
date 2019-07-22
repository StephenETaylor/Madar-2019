You can run the code from `main_s1.py` file in `nn` package

To run the code, you need:
* Installed python libraries specified in requirments.txt
* Arabic word embeddings in .vec format, for example, fasttext from https://fasttext.cc/docs/en/pretrained-vectors.html
    * put the word embedding in the `./emb/ar/` folder
    
* Put files with train, dev and test data into `MADAR-Shared-Task-Subtask-1` folder
    * `MADAR-Corpus-26-train.lm26c`
    * `MADAR-Corpus-26-dev.lm26c`
    * `MADAR-Corpus-26-test.lm26c`
    
    Each line in these files contains:
     * _**Arabic sentence**_
     *  **_LABEL_** (corresponding to a dialect),
     * _**26 floats numbers**_ representing score for each character language model, each dialect has its own language model
     
     * The values are separated by tabulator \tab
     * The test data does not contain the **_LABEL_**
     
     for example:
     
 بالمناسبة ، اسمي هيروش إيجيما .	MSA	0.038870413	0.038708426	0.036623597	0.036556713	0.037177704	0.039430913	0.03795585	0.038743135	0.037666917	0.038870066	0.040305067	0.038132917	0.039700016	0.036438283	0.039093286	0.038653616	0.038263142	0.038167074	0.037563395	0.03926785	0.038904686	0.03604333	0.03931954	0.040455755	0.03787926	0.041209087