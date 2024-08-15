# Using this script we collect the training data for our model
# Sentence transformer models require sentence pair dataset with labels indicating the relationship between sentences
# We collect mapping relationship and concept synonyms on Athena. In wthis way we collect a large-scale sentence-pair dataset with similar sentences

import pandas as pd

# Concept relationship dataframe
df1 = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/xzcode/CDM/data2/CONCEPT_RELATIONSHIP.csv', sep = '\t')

# --------------mapping relationships--------------
df1 = df1[df1['relationship_id'] == 'Mapped from']
df1 = df1[['concept_id_1', 'concept_id_2']]

# Concepts table
df2 = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/xzcode/CDM/data2/CONCEPT.csv', sep = '\t')

# Df into dictionary
id_to_concept_name = df2.set_index('concept_id')['concept_name'].to_dict()
id_to_domain_id = df2.set_index('concept_id')['domain_id'].to_dict()
id_to_vocabulary_id = df2.set_index('concept_id')['vocabulary_id'].to_dict()
id_to_concept_class_id = df2.set_index('concept_id')['concept_class_id'].to_dict()
id_to_standard_concept = df2.set_index('concept_id')['standard_concept'].to_dict()

# Concept id into concept name
df1['concept_name_1'] = df1['concept_id_1'].map(id_to_concept_name)
df1['domain_id_1'] = df1['concept_id_1'].map(id_to_domain_id)
df1['vocabulary_id_1'] = df1['concept_id_1'].map(id_to_vocabulary_id)
df1['concept_class_id_1'] = df1['concept_id_1'].map(id_to_concept_class_id)
df1['standard_concept_1'] = df1['concept_id_1'].map(id_to_standard_concept)

df1['concept_name_2'] = df1['concept_id_2'].map(id_to_concept_name)
df1['domain_id_2'] = df1['concept_id_2'].map(id_to_domain_id)
df1['vocabulary_id_2'] = df1['concept_id_2'].map(id_to_vocabulary_id)
df1['concept_class_id_2'] = df1['concept_id_2'].map(id_to_concept_class_id)
df1['standard_concept_2'] = df1['concept_id_2'].map(id_to_standard_concept)

# Removing Nan
df3 = df1.dropna(subset=['domain_id_1'])
print(df3.shape)

# Concept 1 is standard; concept 2 is not.
df3 = df3[df3['standard_concept_2'] != 'S']
df3 = df3[df3['standard_concept_1'] == 'S']
print(df3.shape)

# Remove situations when concept 1 and 2 are the same
df3 = df3[df3['concept_class_id_1'] != df3['concept_class_id_2']]
pos_sample = df3[['concept_name_1', 'concept_name_2', 'concept_id_1', 'concept_id_2']]
pos_sample = pos_sample[pos_sample['concept_name_1']!=pos_sample['concept_name_2']]

# Now we get all non-standard to standard mapping on Athena
pos_sample.to_csv('pos_samples.csv', index=False)

# --------------concept synonyms--------------
# Load datasets
df = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/xzcode/CDM/data2/CONCEPT_SYNONYM.csv',
                 sep='\t', on_bad_lines = 'skip', low_memory = False)
df2 = pd.read_csv('/home/xz598/data_folder/cmp-jdat-data-folder/xzcode/CDM/data2/CONCEPT.csv', sep = '\t')

# Collect synonym relationships
id_to_concept_name = df2.set_index('concept_id')['concept_name'].to_dict()
id_to_domain_id = df2.set_index('concept_id')['domain_id'].to_dict()
id_to_standarded_or_not = df2.set_index('concept_id')['standard_concept'].to_dict()
df['concept_name'] = df['concept_id'].map(id_to_concept_name)
df['domain_id'] = df['concept_id'].map(id_to_domain_id)
df['standard_concept'] = df['concept_id'].map(id_to_standarded_or_not)

# Do not need to restrict to standard concepts
# There are cases where non-standard concepts with synonyms are mapped to standard ones
# Therefore synonyms of non-standard concepts could be helpful 
# df = df[df['standard_concept'] == 'S'] # do not run this line
df = df.dropna(subset=['domain_id'])
print(df.shape)
# df = df[df['domain_id'].str.contains('Condition|Procedure|Drug')] # if we want to focus on conditions, procedures, and drugs.
print(df.shape)
print(df.domain_id.unique())
df = pd.DataFrame({'concept_name_1': df['concept_name'], 'concept_name_2': df['concept_synonym_name'], 'concept_id_1': df['concept_id']})
df.to_csv('synonyms.csv', index = 'False')


