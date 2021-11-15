#!/usr/bin/python3

import pandas as pd
from datasets import load_dataset

dataset = load_dataset('ade_corpus_v2', "Ade_corpus_v2_classification")
df = pd.DataFrame(dataset["train"])
df.to_csv("ade_causes.csv", index=None) 

dataset = load_dataset('ade_corpus_v2', "Ade_corpus_v2_drug_ade_relation")
df = pd.DataFrame(dataset["train"])
df = df.drop(columns=["indexes"])
df.to_csv("ade_corpus.csv", index=None) 