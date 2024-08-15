# Here we build a sentence-transformer from GatorTron, an encoder-only pretrained model

from sentence_transformers import SentenceTransformer, models, SentencesDataset
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import losses
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers.readers import InputExample
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.readers import InputExample
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

## Step 1: use an existing language model
word_embedding_model = models.Transformer('UFNLP/gatortron-base', max_seq_length=64)

## Step 2: use a pool function over the token embeddings
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

## Join steps 1 and 2 using the modules argument
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Train model using synonyms and mapping relationships on Athena
synonyms = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/drugs/synonyms.csv')
pos_sample = pd.read_csv('/home/xz598/scripts/use_cdm_train_st/drugs/pos_samples.csv')
df = pd.concat([synonyms, pos_sample])

train_examples = []

# df = df.sample(n=100000)
for _, row in df.iterrows():
    concept_name_1 = str(row['concept_name_1'])
    concept_name_2 = str(row['concept_name_2'])
    
    # Skip the row if either concept_name_1 or concept_name_2 is not a string
    if not isinstance(concept_name_1, str) or not isinstance(concept_name_2, str):
        continue
    
    # Skip the row if either concept_name_1 or concept_name_2 is empty
    if not concept_name_1 or not concept_name_2:
        continue
    
    texts = [concept_name_1, concept_name_2]
    example = InputExample(texts=texts)
    train_examples.append(example)
    
train_dataset = SentencesDataset(train_examples, model)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=48)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 1
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.module.fit(train_objectives=[(train_dataloader, train_loss)],
           epochs=num_epochs,
           warmup_steps=warmup_steps,
           show_progress_bar=True)

# save the model
model.save("/home/xz598/scripts/use_cdm_train_st/train_st_from_scrach/model")