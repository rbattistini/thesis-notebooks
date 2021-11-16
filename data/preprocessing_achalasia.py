#!/usr/bin/python3

import sys
import re
import spacy
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def normalize_emotes(df):       
    # EMOTEGOOD :) :-) :] :-] =) =] => :> ^^ ^_^ ^-^ ^o^ : ) (: :'D
    df.original_text = df.original_text.str.replace(
        "\\:\\)+|\\:\\-\\)+|\\:\\]+|\\:\\-\\]+|\\=\\)" +
        "|\\=\\]+|\\=\\>|\\:\\>|\\^\\^|\\^\\_" + 
        "\\^|\\^\\-\\^|\\^o\\^|\\:[[:blank:]]\\) " +
        "|[[:blank:]]\\([[:blank:]]?\\:|\\:\\'D+",
        " emotegood ",
        regex = True
    )
    
    # EMOTEGOOD :d :D :-d :-D =d =D 8d 8D :')
    df.original_text = df.original_text.str.replace(
        "\\:d+|\\:D+|\\:\\-d+|\\:\\-D+|\\=d" + 
        "|\\=D+|8d+|8D+|\\:\\'+\\)+|v\\.v",
        " emotegood ",
        regex = True
    )
    
    # EMOTELOVE <3 :*
    df.original_text = df.original_text.str.replace(
        "\\<3+|\u2764|\u2665|\\:\\*+",
        " emotelove ",
        regex = True
    )
    
    # EMOTEBAD :( :-( :[ :-[ =[ =( : ( ):
    df.original_text = df.original_text.str.replace(
        "\\:\\(+|\\:\\-\\(+|\\:\\[+|\\:\\-\\[+|\\=\\[+|\\=\\(" + 
        "|\\:[[:blank:]]\\(|[[:blank:]]\\([[:blank:]]?\\:",
        " emotebad ",
        regex = True
    )
  
    # EMOTEBAD :'( :-[ D:
    df.original_text = df.original_text.str.replace(
        "\\:\\'+\\(+|\\:\\'\\[|D\\:|\\:\\-\\[",
        " emotebad ",
        regex = True
    )
    
    # EMOTEBAD :| :/ =/ :x :-|
    df.original_text = df.original_text.str.replace(
        "\\:\\|\\:/+|\\=/+|\\:x",
        " emotebad ",
        regex = True
    )
    
    # EMOTEBAD #_# X_X x_x X.X x.x >.< >_< >.> >_>
    df.original_text = df.original_text.str.replace(
        "\\#\\_+\\#|X\\_+X|x\\_+x|X\\.X|x\\.x|>\\.<|>\\_" +
        "<|>\\_+>|>\\.>",
        " emotebad ",
        regex = True
    )
    
    # EMOTEWINK ;) ;-) ;] ;-] ;> ;d ;D ;o
    df.original_text = df.original_text.str.replace(
        "\\;\\)+|\\;\\-\\)+|\\;\\]|\\;\\-\\]|\\;\\>|;d+|;D+|;o",
        " emotewink ",
        regex = True
    )
    
    # EMOTESHOCK O.o o.o O.O o.O O_o o_o O_O o_O etc
    df.original_text = df.original_text.str.replace(
        "O\\.o|o\\.o|O\\.O|o\\.O|O\\_+o|o\\_+o|O\\_+O|o\\_+O|\\:OO" + 
        "|\\=O+|\\-\\.\\-|u\\.u|u\\.\u00F9|\u00F9\\.u|u\\_" + 
        "u|\u00E7\u00E7|\u00E7_+\u00E7|t_+t|\u00F9\\_" + 
        "\u00F9|\u00F9\\.\u00F9|\\:oo+|0\\_+0|\\=\\_" + 
        "\\=|\\.\\_+\\.|\u00F2\u00F2|\u00F2\\_+\u00F2|\\*u+\\*|\\-\\_" +
        "\\-|\u00F9\u00F9|\\-\\,\\-|\\-\\-\\'|\\.\\-\\.|\\'\\-\\'",
        " emoteshock ",
        regex = True
    )

    # EMOTEAMAZE *_* *-* *o* *.*
    df.original_text = df.original_text.str.replace(
        "\\*\\_+\\*|\\*\\-\\*|\\*\\.\\*",
        " emoteamaze ",
        regex = True
    )
    
    # EMOTEJOKE :P :p =P =p XD xD xd d:
    df.original_text = df.original_text.str.replace(
        "\\:P+[^e]|\\:p+[^e]|\\=P+|\\=p+|XD+|xD+|xd+|[[:blank:]]d\\:",
        " emotejoke ",
        regex = True
    )
    
def reduce_lengthening(message):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", message)

def reduce_lengthening_all(messages):
    if isinstance(messages, str):
        messages = [messages]
    reduced_messages = map(lambda m: reduce_lengthening(m), messages)
    return list(reduced_messages)

def spacy_tokenizer(sentence):
    sentence = nlp(sentence)
    tokens = [token.lemma_.lower() for token in sentence 
          if token.is_stop == False 
          and token.is_digit == False
          and token.is_punct == False
          and len(token.text) > 1
          and token.is_ascii == True
          and token.is_alpha == True
          and token.is_quote == False
          and token.is_space == False
          and token.is_currency == False
          and token.like_email == False
          and token.like_num == False
          and token.like_url == False
#           and token.pos_ in ['NOUN', 'ADJ', 'VERB', 'ADV']
         ]
    return tokens

dataset_name = sys.argv[1]

df = pd.read_csv(
    dataset_name,
    dtype={"class" : "category", "original_text" : "object"}
)

df = df.drop(
    columns = ["doc_id", "annotated_text", "creation_date", "score"]
)

df.original_text = df.original_text.str.replace(r'&#10;', ' ')

normalize_emotes(df)

df.original_text = df.original_text.str.lower()

df.original_text = reduce_lengthening_all(df.original_text)

dataset_name = sys.argv[2]
slang_words = pd.read_csv(
    dataset_name
)

dictionary = {}
for row in slang_words.itertuples():
    dictionary[re.escape(row.slang)] = row.phrase
dictionary = {r'\b' + k + r'\b':v for k, v in dictionary.items()}

df.original_text = df.original_text.replace(dictionary, regex=True)

excluded = [
#     "tok2vec",
#     "tagger",
#     "parser",
    "ner",
#     "attribute_ruler",
#     "lemmatizer"
]
nlp = spacy.load("it_core_news_sm", exclude=excluded)

# Handle lemmatization of important terms
nlp.get_pipe('lemmatizer').lookups.get_table("lemma_lookup_noun")["gemelli"] = "gemelli"
nlp.get_pipe('lemmatizer').lookups.get_table("lemma_lookup_noun")["plastico"] = "plastica"

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

sp.save_npz("tdm_achalasia.npz", tdm)

file_name = 'vocabulary_achalasia.sav'
pickle.dump(vocabulary, open(file_name, 'wb'))
loaded_model = pickle.load(open(file_name, 'rb'))

file_name = 'doc_classes_achalasia.sav'
df["class"].to_pickle(file_name)
doc_classes = pd.read_pickle(file_name)

