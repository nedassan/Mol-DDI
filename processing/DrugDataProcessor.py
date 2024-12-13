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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DrugDataProcessor:
    def __init__(self, cache_file='drug_smiles_cache.json'):
        self.cache_file = Path(cache_file)
        self.smiles_cache = self._load_cache()
        self.session = self._create_retry_session()
        
    def _create_retry_session(self):
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.smiles_cache, f)

    def get_smiles_from_pubchem(self, drugbank_id, retry_delay=1):
        if drugbank_id in self.smiles_cache:
            return self.smiles_cache[drugbank_id]

        try:
            time.sleep(retry_delay)
            results = pcp.get_compounds(drugbank_id, 'name')
            
            if results:
                smiles = results[0].isomeric_smiles
                self.smiles_cache[drugbank_id] = smiles
                if len(self.smiles_cache) % 10 == 0:
                    self._save_cache()
                return smiles
            else:
                logger.warning(f"No results found for {drugbank_id}")
                self.smiles_cache[drugbank_id] = None
                return None

        except Exception as e:
            logger.error(f"Error fetching SMILES for {drugbank_id}: {str(e)}")
            return None

    def smiles_to_smarts(self, smiles):
        """Convert SMILES to SMARTS representation"""
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmarts(mol)
        return None

    def process_drug_pairs(self, input_file, batch_size=1000):
        logger.info(f"Processing {input_file}")
        
        chunks = pd.read_csv(
            input_file,
            delimiter='\t',
            compression='gzip' if input_file.endswith('.gz') else None,
            names=['Drug1_ID', 'Drug2_ID'],
            chunksize=batch_size
        )

        processed_chunks = []
        
        for chunk in chunks:
            tqdm.pandas(desc="Processing Drug1")
            chunk['Drug1_SMILES'] = chunk['Drug1_ID'].progress_apply(self.get_smiles_from_pubchem)
            
            tqdm.pandas(desc="Processing Drug2")
            chunk['Drug2_SMILES'] = chunk['Drug2_ID'].progress_apply(self.get_smiles_from_pubchem)
            
            chunk = chunk.dropna(subset=['Drug1_SMILES', 'Drug2_SMILES'])
            
            if len(chunk) == 0:
                continue
                
            tqdm.pandas(desc="Converting to SMARTS")
            chunk['Drug1_SMARTS'] = chunk['Drug1_SMILES'].progress_apply(self.smiles_to_smarts)
            chunk['Drug2_SMARTS'] = chunk['Drug2_SMILES'].progress_apply(self.smiles_to_smarts)
            
            
            processed_chunks.append(chunk)
            

        if not processed_chunks:
            logger.warning("No valid data processed")
            return pd.DataFrame()
            
        final_data = pd.concat(processed_chunks, ignore_index=True)
        
        final_data = final_data.dropna(subset=[
            'Drug1_SMARTS', 'Drug2_SMARTS'
        ])
        
        output_dir = Path('processed_data')
        output_dir.mkdir(exist_ok=True)
        
        final_data.to_csv(
            output_dir / 'processed_drug_pairs.tsv.gz',
            sep='\t',
            compression='gzip',
            index=False
        )
        
        
        logger.info("Processing completed. Files saved in 'processed_data' directory.")
        
        return final_data


if __name__ == "__main__":
    processor = DrugDataProcessor()
    
    input_file = 'data/ChCh-Miner_durgbank-chem-chem.tsv.gz'
    processed_data = processor.process_drug_pairs(input_file)
    
    print("\nProcessing Summary:")
    print(f"Total valid pairs processed: {len(processed_data)}")
    print("\nSample of processed data:")
    print(processed_data.head())
