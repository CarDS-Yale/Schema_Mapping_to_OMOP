### Python code (tutorial) for **A Novel Sentence Transformer-based Natural Language Processing Approach for Schema Mapping of Electronic Health Records to the OMOP Common Data Model**

[link to the tutorial](https://github.com/CarDS-Yale/Schema_Mapping_to_OMOP/blob/main/tutorial.ipynb)

Mapping electronic health records (EHR) data to common data models (CDMs) enables the standardization of clinical records, enhancing interoperability and enabling large-scale, multi-centered clinical investigations. Using 2 large publicly available datasets, we developed transformer-based natural language processing models to map medication-related concepts from the EHR at a large and diverse healthcare system to standard concepts in OMOP CDM. We validated the model outputs against standard concepts manually mapped by clinicians. Our best model reached out-of-box accuracies of 96.5% in mapping the 200 most common drugs and 83.0% in mapping 200 random drugs in the EHR. For these tasks, this model outperformed a state-of-the-art large language model (SFR-Embedding-Mistral, 89.5% and 66.5% in accuracy for the two tasks), a widely-used software for schema mapping (Usagi, 90.0% and 70.0% in accuracy), and direct string match (7.5% and 7.5% accuracy). Transformer-based deep learning models outperform existing approaches in the standardized mapping of EHR elements and can facilitate an end-to-end automated EHR transformation pipeline.  

The study was supported by funding from the National Heart, Lung, and Blood Institute under the grant **R01HL167858**

**Please cite this study as:**
```
@article{zhou2024omop,
  title={A Novel Sentence Transformer-based Natural Language Processing Approach for Schema Mapping of Electronic Health Records to the OMOP Common Data Model},
  author={Zhou, Xinyu and Dhingra, Lovedeep Singh and Aminorroaya, Arya and Adejumo, Philip and Khera, Rohan},
  journal={medRxiv},
  pages={2024--03},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```