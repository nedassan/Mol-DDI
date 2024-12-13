import torch
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader, Data
from sklearn.manifold import TSNE
from torch import nn
from MoltoGraph.MoleculeToGraph import MoleculeToGraph
from models.GAT import GNN as GAT
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from rdkit import Chem 
from rdkit.Chem import Draw
from PIL import Image
from rdkit.Chem import Descriptors


class MoleculeEmbeddingVisualizer:
    def __init__(self, gnn_model, drug_smiles_file, radius=2, bit_length=1024, n_clusters=3, device = None):
        self.gnn_model = gnn_model
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.gnn_model.to(self.device)
        self.gnn_model.eval()
        self.drug_smiles_file = drug_smiles_file
        self.radius = radius
        self.bit_length = bit_length
        self.n_clusters = n_clusters
        self.converter = MoleculeToGraph(radius=radius, bit_length=bit_length)

        with open(drug_smiles_file, 'r') as f:
            self.drug_smiles_dict = json.load(f)
    
    def get_embeddings(self):
        embeddings = []
        drug_ids = []
        for drug_id, smiles in self.drug_smiles_dict.items():
            if not smiles:
                continue  
            graph_data, _ = self.converter.smiles_to_graph_data(smiles)
            drug_ids.append(drug_id)
            with torch.no_grad():
                try:
                    embedding = self.gnn_model(graph_data)
                except Exception as e:
                    # print(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
                    drug_ids.pop(0)
                    continue
            embeddings.append(embedding.cpu().numpy())

        embeddings = np.array(embeddings)
        return embeddings, drug_ids
    
    def visualize_clusters(self, embeddings, drug_ids, cluster_labels, num_sampled = 5):
        sampled_molecules = {}
        unique_clusters = np.unique(cluster_labels)

        for cluster in unique_clusters:
            cluster_indices = np.where(cluster_labels == cluster)[0]
            sampled_indices = np.random.choice(cluster_indices, size = min(num_sampled, len(cluster_indices)), replace = False)
            sampled_molecules[cluster] = [drug_ids[idx] for idx in sampled_indices]

        return sampled_molecules

    def cluster_and_visualize(self, num_mols_sampled = 5):
        embeddings, drug_ids = self.get_embeddings()
        embeddings = embeddings.squeeze(1)
        print(embeddings.shape)
        pca = PCA(n_components=50)
        pca_embeddings = pca.fit_transform(embeddings)

        tsne = TSNE(n_components=2, random_state=42)
        tsne_embeddings = tsne.fit_transform(pca_embeddings)

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tsne_embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=cluster_labels, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Molecule Embeddings Visualization')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')

        plt.show()

        sampled_molecules = self.visualize_clusters(embeddings, drug_ids, cluster_labels, num_mols_sampled)

        cluster_imgs = []
        for cluster, molecules in sampled_molecules.items():
            chem_mols = [Chem.MolFromSmiles(self.drug_smiles_dict[drug_id]) for drug_id in molecules]
            img = Draw.MolsToGridImage(chem_mols, molsPerRow = num_mols_sampled, subImgSize = (200, 200), legends = [f"Cluster {cluster}\n{drug_id}" for drug_id in molecules])
            cluster_imgs.append((cluster, img))

        widths, heights = zip(*(img.size for _, img in cluster_imgs))
        total_width = max(widths)
        total_height = sum(heights)

        combined_image = Image.new("RGB", (total_width, total_height))
        y_offset = 0
        for _, img in sorted(cluster_imgs, key=lambda x: x[0]):
            combined_image.paste(img, (0, y_offset))
            y_offset += img.size[1]

        combined_image.show()


if __name__ == '__main__':

    gnn = GAT(
        input_dim = 1024, 
        hidden_dim = 256,
        output_dim = 128,
        activation_fn = torch.nn.GELU(),
        pooling_fn = global_mean_pool,
        dropout_rate = 0.2,
        num_heads = 4,
        edge_dim = 6
    )

    full_state_dict = torch.load('gat_lstm_sim_pen/model.pt', weights_only = True)

    gnn_state_dict = {
        k.replace('gnn.', ''): v 
        for k, v in full_state_dict.items() 
        if k.startswith('gnn.')
    }

    gnn.load_state_dict(gnn_state_dict)
    
    drug_smiles_file = 'drug_smiles_cache.json'
    
    visualizer = MoleculeEmbeddingVisualizer(gnn, drug_smiles_file, n_clusters=10)
    visualizer.cluster_and_visualize(num_mols_sampled = 10)