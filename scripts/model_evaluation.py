# Here we evaluate and compare the models' performance in mapping drug concepts from Yale's EHR to OMOP CDM 
# We presented how we generate model outputs for medications

from sentence_transformers import SentenceTransformer, models, SentencesDataset
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import losses
import torch
import pandas as pd
import re

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data from athena
concepts = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/evaluation/CONCEPT.csv', sep = '\t')
synonyms = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/evaluation/CONCEPT_SYNONYM.csv', sep = '\t')

# If any data preprocessing needed
def data_preprocessing(text):
# use any data preprocessing that is reasonable
    return text

# Now we load data from YNHHS
df_ehr_full = pd.read_csv('/home/xz598/data_folder/2356781_CarDS_Aim_1_Meds.txt',
                 sep='\t', on_bad_lines = 'skip', low_memory = False)
df_ehr_full = df_ehr_full[~df_ehr_full['ORDER_DESCRIPTION'].str.lower().isin(concepts['concept_name'].str.lower()) & ~df_ehr_full['ORDER_DESCRIPTION'].str.lower().isin(synonyms['concept_synonym_name'].str.lower())]
value_counts = df_ehr_full['ORDER_DESCRIPTION'].value_counts()
sorted_counts = value_counts.sort_values(ascending=False) 
df_ehr_top500 = sorted_counts.head(500)
df_ehr_top500 = pd.DataFrame({'ORDER_DESCRIPTION': df_ehr_top500.index, 'Count': df_ehr_top500.values})
df_ehr_top500['processed'] = df_ehr_top500['ORDER_DESCRIPTION'].map(data_preprocessing)

# Standard medication concepts from OMOP CDM
cdm_c = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/evaluation/CONCEPT.csv', sep = '\t')
cdm_c = cdm_c[cdm_c['standard_concept'] == 'S'] # standard concepts
cdm_c = cdm_c[cdm_c['domain_id'] == 'Drug'] # only include drugs
cdm_c.reset_index(inplace=True)

# Set more models if you want
model1 = SentenceTransformer("all-mpnet-base-v2") # the best publicly available model on 'sentence transformers' huggingface repo
model2 = SentenceTransformer("/home/xz598/scripts/use_cdm_train_st/model_for_all_domains/model") # we pretrained the model above using concept mapping relationships

def find_best_match(model, colname, ehrcol):
    embedding_cdm_c = model.encode(cdm_c['concept_name']) # encode all standard concept names into latent space 
    embeddings_ehr_top500= model.encode(df_ehr_top500[ehrcol]) # encode the 500 most common drugs into latent space as well
    similarity_matrix_top500 = cosine_similarity(embeddings_ehr_top500, embedding_cdm_c) # find best match using cosine similarity
    
    # find the closet match
    most_similar_indices_top500 = np.argmax(similarity_matrix_top500, axis=1)
    most_similar_sentences_top500 = cdm_c['concept_name'].iloc[most_similar_indices_top500]
    df_ehr_top500[colname] = most_similar_sentences_top500.values
    
    # find the similarity score of the closet match
    most_similar_values_top500 = np.max(similarity_matrix_top500, axis=1)
    df_ehr_top500[f'similarity_score_{colname}'] = most_similar_values_top500
    
find_best_match(model1, 'all-mpnet-base-v2', 'ORDER_DESCRIPTION')
find_best_match(model2, 'all-mpnet-base-v2-trained', 'ORDER_DESCRIPTION')

df_ehr_top500.to_csv('/home/xz598/scripts/use_cdm_train_st/evaluation/top_500_drugs.csv', index = False)
