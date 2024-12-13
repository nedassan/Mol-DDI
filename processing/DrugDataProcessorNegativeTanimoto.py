import random
import pandas as pd
import json
from rdkit import Chem
from pathlib import Path
from DrugDataProcessor import DrugDataProcessor
from rdkit.Chem import AllChem
from rdkit import DataStructs

class DrugDataProcessorWithNegativeTanimoto(DrugDataProcessor):
    def compute_tanimoto_similarity(self, smiles_list):
        
        print("Computing Tanimoto similarity matrix...")
        fingerprints = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius=2) for smiles in smiles_list]
        num_drugs = len(fingerprints)
        similarity_matrix = []

        for i in range(num_drugs):
            row = []
            for j in range(i + 1, num_drugs):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                row.append((i, j, sim))
            similarity_matrix.extend(row)
        
        return similarity_matrix

    def generate_negative_samples(self, positive_pairs, smiles_dict, num_negatives=None):
        print('Processing positive samples')
        all_drugs = list(set(positive_pairs['Drug1_ID']).union(set(positive_pairs['Drug2_ID'])))
        drug_smiles = [smiles_dict[drug_id] for drug_id in all_drugs if drug_id in smiles_dict]
        similarity_matrix = self.compute_tanimoto_similarity(drug_smiles)
        
        positive_pairs_set = set((min(d1, d2), max(d1, d2)) for d1, d2 in zip(positive_pairs['Drug1_ID'], positive_pairs['Drug2_ID']))

        negative_candidates = [
            (all_drugs[i], all_drugs[j], sim)
            for i, j, sim in similarity_matrix
            if (all_drugs[i], all_drugs[j]) not in positive_pairs_set
        ]
        negative_candidates = sorted(negative_candidates, key=lambda x: -x[2]) 

        num_negatives = num_negatives or len(positive_pairs)
        negative_pairs = set((min(d1, d2), max(d1, d2)) for d1, d2, _ in negative_candidates[:num_negatives])
        
        print(f"Generated {len(negative_pairs)} negative samples.")
        return pd.DataFrame(negative_pairs, columns=['Drug1_ID', 'Drug2_ID'])

    def save_combined_samples(self, positive_samples, negative_samples, smiles_dict):
        print("Combining positive and negative samples...")
        
        negative_samples['Drug1_SMILES'] = negative_samples['Drug1_ID'].apply(lambda x: smiles_dict.get(x, None))
        negative_samples['Drug2_SMILES'] = negative_samples['Drug2_ID'].apply(lambda x: smiles_dict.get(x, None))
        
        negative_samples['Drug1_SMARTS'] = negative_samples['Drug1_SMILES'].apply(self.smiles_to_smarts)
        negative_samples['Drug2_SMARTS'] = negative_samples['Drug2_SMILES'].apply(self.smiles_to_smarts)
        
        negative_samples = negative_samples.dropna(subset=['Drug1_SMILES', 'Drug2_SMILES', 'Drug1_SMARTS', 'Drug2_SMARTS'])
        
        combined_samples = pd.concat([positive_samples, negative_samples], ignore_index=True)
        combined_samples = combined_samples.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output_dir = Path('processed_data')
        output_dir.mkdir(exist_ok=True)
        combined_samples.to_csv(output_dir / 'combined_similarity_score_drug_pairs_with_smiles_smarts.tsv.gz', sep='\t', compression='gzip', index=False)
        
        print(f"Combined samples saved to {output_dir / 'combined_similarity_score_drug_pairs_with_smiles_smarts.tsv.gz'}")

    def process_drug_pairs_with_negatives(self, input_file, smiles_dict, num_negatives_per_positive=1, batch_size=1000):
        print("Processing positive samples...")
        
        positive_samples = self.process_drug_pairs(input_file, batch_size)
        positive_samples['Label'] = 1
        
        negative_samples = self.generate_negative_samples(positive_samples, smiles_dict, num_negatives=num_negatives_per_positive * len(positive_samples))
        negative_samples['Label'] = 0 
        
        print("Processing completed for both positive and negative samples.")
        
        self.save_combined_samples(positive_samples, negative_samples, smiles_dict)

        return positive_samples, negative_samples


if __name__ == "__main__":
    processor = DrugDataProcessorWithNegativeTanimoto()
    
    input_file = 'data/ChCh-Miner_durgbank-chem-chem.tsv.gz'
    smiles_dict_file = 'drug_smiles_cache.json'
    
    with open(smiles_dict_file, 'r') as f:
        smiles_dict = json.load(f)
    
    positive_samples, negative_samples = processor.process_drug_pairs_with_negatives(input_file, smiles_dict)
    
    print("\nProcessing Summary:")
    print(f"Total valid positive pairs processed: {len(positive_samples)}")
    print(f"Total valid negative pairs processed: {len(negative_samples)}")
    print("\nSample of positive data:")
    print(positive_samples.head())
    print("\nSample of negative data:")
    print(negative_samples.head())