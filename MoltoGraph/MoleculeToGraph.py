import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import torch
from torch_geometric.data import Data

class MoleculeToGraph:

    def __init__(self, radius=2, bit_length=1024):

        self.radius = radius
        self.bit_length = bit_length
        self.morgan_gen = GetMorganGenerator(radius=radius, fpSize=bit_length)
    
    def _get_atom_features(self, mol, atom_idx):

        atom_fingerprint = np.zeros((self.bit_length,), dtype=np.float32)
        info = {}

        fp = self.morgan_gen.GetFingerprintAsNumPy(mol, fromAtoms=[atom_idx])

        atom_fingerprint = fp.astype(np.float32)
            
        return atom_fingerprint
    
    def _get_bond_features(self, bond):

        bond_type = bond.GetBondTypeAsDouble()
        bond_features = np.array([
            bond_type == 1.0,  # single bond
            bond_type == 2.0,  # double bond
            bond_type == 3.0,  # triple bond
            bond_type == 1.5,  # aromatic bond
            bond.IsInRing(),  
            bond.GetIsConjugated() 
        ], dtype=np.float32)
        return bond_features
    
    def smiles_to_graph_data(self, smiles):

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        
        num_atoms = mol.GetNumAtoms()
        
        x = []
        edge_indices = []
        edge_attrs = []
        
        for atom_idx in range(num_atoms):
            atom_features = self._get_atom_features(mol, atom_idx)
            x.append(atom_features)
        
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])
            
            bond_features = self._get_bond_features(bond)
            edge_attrs.extend([bond_features, bond_features])
        
        x = torch.FloatTensor(np.array(x, dtype=np.float32))
        edge_index = torch.LongTensor(np.array(edge_indices, dtype=np.int64)).t().contiguous()
        edge_attr = torch.FloatTensor(np.array(edge_attrs, dtype=np.float32))
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_atoms
        )
        
        mol_info = {
            'num_atoms': num_atoms,
            'num_bonds': mol.GetNumBonds(),
            'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
            'fingerprint_radius': self.radius,
            'fingerprint_length': self.bit_length
        }
        
        return data, mol_info

    def process_batch(self, smiles_list):

        results = []
        for smiles in smiles_list:
            try:
                graph_data, mol_info = self.smiles_to_graph_data(smiles)
                results.append((graph_data, mol_info))
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {str(e)}")
        return results

def analyze_graph_data(graph_data, mol_info):

    print("Graph Analysis:")
    print(f"Number of nodes: {graph_data.num_nodes}")
    print(f"Number of edges: {graph_data.edge_index.size(1)}")
    print(f"Node feature dimension: {graph_data.x.size(1)}")
    print(f"Edge feature dimension: {graph_data.edge_attr.size(1)}")
    
    print("\nMolecule Information:")
    for key, value in mol_info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    smiles_list = [
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        "CCO",
        "C1=CC=CC=C1"
    ]
    
    converter = MoleculeToGraph(radius=2, bit_length=1024)

    results = converter.process_batch(smiles_list)
    for i, (graph_data, mol_info) in enumerate(results):
        print(f"\nMolecule {i+1}:")
        analyze_graph_data(graph_data, mol_info)
