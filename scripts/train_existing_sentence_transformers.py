# Here we train an existing sentence transformers model

from sentence_transformers import SentenceTransformer, models, SentencesDataset
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import losses
import torch
import pandas as pd
import os
from sentence_transformers.readers import InputExample

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

# Load the model
model = SentenceTransformer('all-mpnet-base-v2')

# If you have more than one GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count() , "GPUs!")
    model = nn.DataParallel(model)

# Then move the model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Train the model using synonyms and mapping relationships on Athena
synonyms = pd.read_csv('synonyms.csv')
pos_sample = pd.read_csv('pos_samples2.csv')
training_data = pd.concat([synonyms, pos_sample])
print(training_data.head())
train_examples = []

for _, row in training_data.iterrows():
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

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=96)
train_loss = losses.MultipleNegativesRankingLoss(model=model)
num_epochs = 10
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

model.module.fit(train_objectives=[(train_dataloader, train_loss)],
           epochs=num_epochs,
           warmup_steps=warmup_steps,
           show_progress_bar=True)

torch.save(model.module.state_dict(), "/home/xz598/scripts/use_cdm_train_st/model_for_all_domains/model")

