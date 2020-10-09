# LDA with gibbs sampling
## About input data

### Vocabulary data
Create a vocabulary file like sample/cran_vocabulary_data.txt and save it as .txt file

### Document data
Create a document file like sample/cran_document_data.pickle and save it as .pickle file


Rewrite the words to their id like the following.

[
[the,cat,under,the,table,is....(document0)]
[(document1)]
[(document2)]
]
↓
[
[10,1,2,1,3,4....(document0)]
[(document1)]
[(document2)]
]
