import pandas as pd
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from MoltoGraph.MoleculeToGraph import MoleculeToGraph
from MoltoGraph.MoleculeToGraphWithChirality import MoleculeToGraphWithChirality
import torch
from torch_geometric.data import Data, Batch
import os
import tempfile
import shutil
import gzip
from typing import Iterator, Tuple, Optional, List

def custom_collate(batch):

    if not batch:  
        return None
        
    drug1_graphs = [item[0] for item in batch]
    drug2_graphs = [item[1] for item in batch]
    num_atoms1 = [item[2] for item in batch]
    num_atoms2 = [item[3] for item in batch]
    labels = torch.tensor([item[4] for item in batch])
    
    batched_drug1 = Batch.from_data_list(drug1_graphs)
    batched_drug2 = Batch.from_data_list(drug2_graphs)
    
    return batched_drug1, batched_drug2, num_atoms1, num_atoms2, labels

class StreamingDrugDataset(IterableDataset):
    def __init__(self, tsv_file: str, cache_dir: str = './cache', batch_size: int = 1000, 
                 start_idx: int = 0, end_idx: Optional[int] = None, rebuild_cache: bool = False):
        self.tsv_file = tsv_file
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        
        self.temp_dir = tempfile.mkdtemp(prefix='drug_data_temp_')
        
        if rebuild_cache and os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        
        os.makedirs(cache_dir, exist_ok=True)

    def __del__(self):
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _process_smiles(self, drug1_smiles: str, drug2_smiles: str, label: int) -> Optional[Tuple[Data, Data, int]]:
        try:
            graph_converter = MoleculeToGraph(radius=2, bit_length=1024)
            drug1_graph, _ = graph_converter.smiles_to_graph_data(drug1_smiles)
            drug2_graph, _ = graph_converter.smiles_to_graph_data(drug2_smiles)
            
            drug1_graph.num_nodes = drug1_graph.x.size(0)
            drug2_graph.num_nodes = drug2_graph.x.size(0)
            
            return drug1_graph, drug2_graph, drug1_graph.num_nodes, drug2_graph.num_nodes, int(label)
        except Exception as e:
            print(f"Error converting SMILES: {e}")
            return None

    def _get_cache_path(self, idx: int) -> str:
        return os.path.join(self.cache_dir, f'batch_{idx}.pt')

    def _get_temp_cache_path(self, idx: int) -> str:
        return os.path.join(self.temp_dir, f'batch_{idx}.pt')

    def _cache_exists(self, idx: int) -> bool:
        cache_path = self._get_cache_path(idx)
        return os.path.exists(cache_path)

    def _save_to_cache(self, batch_data: list, idx: int):
        if not batch_data: 
            return
            
        temp_path = self._get_temp_cache_path(idx)
        final_path = self._get_cache_path(idx)
        
        try:
            processed_data = []
            for drug1_graph, drug2_graph, num_atoms1, num_atoms2, label in batch_data:
                drug1_dict = {
                    'x': drug1_graph.x.cpu(),
                    'edge_index': drug1_graph.edge_index.cpu(),
                    'edge_attr': drug1_graph.edge_attr.cpu() if drug1_graph.edge_attr is not None else None,
                    'num_nodes': drug1_graph.num_nodes
                }
                
                drug2_dict = {
                    'x': drug2_graph.x.cpu(),
                    'edge_index': drug2_graph.edge_index.cpu(),
                    'edge_attr': drug2_graph.edge_attr.cpu() if drug2_graph.edge_attr is not None else None,
                    'num_nodes': drug2_graph.num_nodes
                }
                
                processed_data.append((drug1_dict, drug2_dict, num_atoms1, num_atoms2, label))
            
            torch.save(processed_data, temp_path)
            
            shutil.move(temp_path, final_path)
            
        except Exception as e:
            print(f"Error saving cache: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _load_from_cache(self, idx: int) -> List[Tuple[Data, Data, int]]:
        cache_path = self._get_cache_path(idx)
        try:
            batch_data = torch.load(cache_path, map_location='cpu', weights_only = True)
            processed_batch = []
            
            for drug1_dict, drug2_dict, num_atoms1, num_atoms2, label in batch_data:
                drug1_graph = Data(
                    x=drug1_dict['x'],
                    edge_index=drug1_dict['edge_index'],
                    edge_attr=drug1_dict['edge_attr'],
                    num_nodes=drug1_dict['x'].size(0)  
                )
                
                drug2_graph = Data(
                    x=drug2_dict['x'],
                    edge_index=drug2_dict['edge_index'],
                    edge_attr=drug2_dict['edge_attr'],
                    num_nodes=drug2_dict['x'].size(0)  
                )
                
                processed_batch.append((drug1_graph, drug2_graph, num_atoms1, num_atoms2, label))
                
            return processed_batch
        except Exception as e:
            print(f"Error loading cache {cache_path}: {e}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None

    def _process_chunk(self, chunk: pd.DataFrame) -> list:
        processed_data = []
        for _, row in chunk.iterrows():
            try:
                result = self._process_smiles(
                    row['Drug1_SMILES'],
                    row['Drug2_SMILES'],
                    row['Label']
                )
                if result is not None:
                    processed_data.append(result)
            except Exception as e:
                print(f"Error processing chunk row: {e}")
                continue
        return processed_data

    def __iter__(self) -> Iterator[Tuple[Data, Data, int, int, int]]:
        worker_info = torch.utils.data.get_worker_info()
        chunk_size = self.batch_size
        
        if worker_info is not None:
            per_worker = int(np.ceil((self.end_idx - self.start_idx) / worker_info.num_workers))
            worker_id = worker_info.id
            start_idx = self.start_idx + worker_id * per_worker
            end_idx = min(start_idx + per_worker, self.end_idx)
            chunk_size = chunk_size // worker_info.num_workers
        else:
            start_idx = self.start_idx
            end_idx = self.end_idx

        try:
            chunk_iterator = pd.read_csv(
                self.tsv_file,
                chunksize=chunk_size,
                skiprows=range(1, start_idx + 1) if start_idx > 0 else None,
                nrows=(end_idx - start_idx) if end_idx is not None else None,
                usecols=['Drug1_SMILES', 'Drug2_SMILES', 'Label'],
                sep='\t',
                compression='gzip'
            )

            current_batch = start_idx // chunk_size
            for chunk in chunk_iterator:
                batch_data = None
                if self._cache_exists(current_batch):
                    batch_data = self._load_from_cache(current_batch)
                
                if batch_data is None: 
                    batch_data = self._process_chunk(chunk)
                    self._save_to_cache(batch_data, current_batch)
                
                for item in batch_data:
                    yield item
                    
                current_batch += 1
                
        except Exception as e:
            print(f"Error in iterator: {e}")
            raise

def create_data_loaders(tsv_file: str, batch_size: int = 32, cache_dir: str = './cache', 
                       train_split: float = 0.9, rebuild_cache: bool = False):
    with gzip.open(tsv_file, 'rt') as f:  
        total_rows = sum(1 for _ in f) - 1 
    
    train_size = int(total_rows * train_split)
    
    train_dataset = StreamingDrugDataset(
        tsv_file=tsv_file,
        cache_dir=os.path.join(cache_dir, 'train'),
        batch_size=1000,
        start_idx=0,
        end_idx=train_size,
        rebuild_cache=rebuild_cache
    )
    
    test_dataset = StreamingDrugDataset(
        tsv_file=tsv_file,
        cache_dir=os.path.join(cache_dir, 'test'),
        batch_size=1000,
        start_idx=train_size,
        end_idx=total_rows,
        rebuild_cache=rebuild_cache
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,
        persistent_workers=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        collate_fn=custom_collate,
        persistent_workers=True
    )
    
    return train_loader, test_loader
