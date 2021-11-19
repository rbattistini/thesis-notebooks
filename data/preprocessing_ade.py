#!/usr/bin/python3

import spacy
import scispacy
import pandas as pd
from datasets import load_dataset

def spacy_tokenizer(sentence):
    sentence = nlp(sentence)
    tokens = [token.lemma_.lower() for token in sentence 
          if token.is_stop == False 
          and token.is_punct == False
          and len(token.text) > 1
          and token.is_ascii == True
          and token.is_alpha == True
          and token.is_quote == False
          and token.is_space == False
          and token.like_num == False
          and token.like_url == False
         ]
    return tokens

dataset = load_dataset('ade_corpus_v2', "Ade_corpus_v2_classification")
df = pd.DataFrame(dataset["train"])
df.to_csv("ade_causes.csv", index=None) 

dataset = load_dataset('ade_corpus_v2', "Ade_corpus_v2_drug_ade_relation")
df = pd.DataFrame(dataset["train"])
df = df.drop(columns=["indexes"])
df.to_csv("ade_corpus.csv", index=None)

# To download the pipeline
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
excluded = [
#     "tok2vec",
#     "tagger",
#     "parser",
    "ner",
#     "attribute_ruler",
#     "lemmatizer"
]
nlp = spacy.load("en_core_sci_sm", exclude=excluded)

vect = CountVectorizer(
    lowercase = True,
    strip_accents = "ascii",
    tokenizer = spacy_tokenizer,
    min_df = 0.002
)

dtm = vect.fit_transform(df.original_text)
vocabulary = vect.vocabulary_
ord_terms = vect.get_feature_names()

tdm = dtm.transpose()

sp.save_npz("tdm_ade.npz", tdm)

file_name = 'vocabulary_ade.sav'
pickle.dump(vocabulary, open(file_name, 'wb'))
loaded_model = pickle.load(open(file_name, 'rb'))
