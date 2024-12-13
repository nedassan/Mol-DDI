import random
import pandas as pd
import pubchempy as pcp
from rdkit import Chem
import time
from pathlib import Path
import json
import logging
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from DrugDataProcessor import DrugDataProcessor

class DrugDataProcessorWithNegatives(DrugDataProcessor):
    def generate_negative_samples(self, positive_pairs, num_negatives=None):
        print("Generating negative samples")
        
        all_drugs = set(positive_pairs['Drug1_ID']).union(set(positive_pairs['Drug2_ID']))
        all_drugs = list(all_drugs)
        
        positive_pairs_set = set((min(d1, d2), max(d1, d2)) for d1, d2 in zip(positive_pairs['Drug1_ID'], positive_pairs['Drug2_ID']))
        
        negative_pairs = set()
        
        while len(negative_pairs) < (num_negatives or len(positive_pairs)):
            drug1, drug2 = random.sample(all_drugs, 2)
            pair = (min(drug1, drug2), max(drug1, drug2))
            
            if pair not in positive_pairs_set and pair not in negative_pairs:
                negative_pairs.add(pair)
        
        print(f"Generated {len(negative_pairs)} negative samples.")
        
        negative_samples_df = pd.DataFrame(negative_pairs, columns=['Drug1_ID', 'Drug2_ID'])
        
        return negative_samples_df
    
    def process_drug_pairs_with_negatives(self, input_file, num_negatives_per_positive=1, batch_size=1000):
        print("Processing positive samples...")
        
        positive_samples = self.process_drug_pairs(input_file, batch_size)
        positive_samples['Label'] = 1
        
        print("Generating negative samples...")
        negative_samples = self.generate_negative_samples(positive_samples, num_negatives=num_negatives_per_positive * len(positive_samples))
        negative_samples['Label'] = 0 
        
        print("Processing negative samples...")
        tqdm.pandas(desc="Processing Negative Drug1")
        negative_samples['Drug1_SMILES'] = negative_samples['Drug1_ID'].progress_apply(self.get_smiles_from_pubchem)
        
        tqdm.pandas(desc="Processing Negative Drug2")
        negative_samples['Drug2_SMILES'] = negative_samples['Drug2_ID'].progress_apply(self.get_smiles_from_pubchem)
        
        negative_samples = negative_samples.dropna(subset=['Drug1_SMILES', 'Drug2_SMILES'])
        
        if len(negative_samples) == 0:
            print("No valid negative samples processed.")
            return positive_samples, pd.DataFrame()
        
        tqdm.pandas(desc="Converting Negative to SMARTS")
        negative_samples['Drug1_SMARTS'] = negative_samples['Drug1_SMILES'].progress_apply(self.smiles_to_smarts)
        negative_samples['Drug2_SMARTS'] = negative_samples['Drug2_SMILES'].progress_apply(self.smiles_to_smarts)
        
        negative_samples = negative_samples.dropna(subset=['Drug1_SMARTS', 'Drug2_SMARTS'])
        
        print("Processing completed for both positive and negative samples.")
        
        combined_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        combined_samples = combined_samples.sample(frac=1, random_state=42).reset_index(drop=True)

        output_dir = Path('processed_data')
        output_dir.mkdir(exist_ok=True)
        
        combined_samples.to_csv(output_dir / 'combined_drug_pairs.tsv.gz', sep='\t', compression='gzip', index=False)

        
        return positive_samples, negative_samples, combined_samples



if __name__ == "__main__":
    processor = DrugDataProcessorWithNegatives()
    
    input_file = 'data/ChCh-Miner_durgbank-chem-chem.tsv.gz'
    
    positive_samples, negative_samples, combined_samples = processor.process_drug_pairs_with_negatives(input_file)
    
    print("\nProcessing Summary:")
    print(f"Total valid positive pairs processed: {len(positive_samples)}")
    print(f"Total valid negative pairs processed: {len(negative_samples)}")
    print(f"Total valid combined pairs processed: {len(combined_samples)}")
    print("\nSample of positive data:")
    print(positive_samples.head())
    print("\nSample of negative data:")
    print(negative_samples.head())
    print("\nSample of combined data:")
    print(combined_samples.head())
