import os
import json
import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
from process import create_data_loaders
from models.GAT import GNN as GAT
from models.GCN import GNN as GCN
from models.GCN_no_edge_attr import GNN_no_edge_attr as GCN_no_edge_attr
from models.DDI_prediction import FFNN, FFNNAttn, LSTM_DDI, GRU_DDI
import matplotlib.pyplot as plt
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from evaluation import evaluate_model


def save_model_and_history(model, history, save_dir='model_checkpoints'):
    """
    Save the model state dict and training history.
    
    Args:
        model: The trained PyTorch model
        history (dict): Dictionary containing training metrics
        save_dir (str): Directory to save the model and history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(model.state_dict(), model_path)
    
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        serializable_history = {
            'epoch_losses': [float(x) for x in history['epoch_losses']],
            'epoch_accuracies': [float(x) for x in history['epoch_accuracies']]
        }
        json.dump(serializable_history, f, indent=4)
    
    print(f"Model saved to {model_path}")
    print(f"Training history saved to {history_path}")

def plot_training_curves(history, save_path=None):

    plt.figure(figsize=(10, 6))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#8884d8' 
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color1)
    epochs = range(1, len(history['epoch_losses']) + 1)
    line1 = ax1.plot(epochs, 
                     history['epoch_losses'], 
                     color=color1, 
                     marker='o',
                     label='Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) 
    
    ax2 = ax1.twinx()
    color2 = '#82ca9d' 
    ax2.set_ylabel('Accuracy', color=color2)
    line2 = ax2.plot(epochs, 
                     history['epoch_accuracies'], 
                     color=color2, 
                     marker='o',
                     label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax2.set_ylim([0, 1])
    
    ax1.grid(True, alpha=0.3)
    
    plt.title('Training Curve', pad=20)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, num_epochs=10, device=None, save_dir='model_checkpoints', save_model = False, plot_training_curve = False):
    
    optimizer = optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 1e-2)
    loss_fn = torch.nn.BCELoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.train()
    model.to(device)
    
    history = {
        'epoch_losses': [],
        'epoch_accuracies': []
    }
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, datapoint in enumerate(progress_bar):
            molecule1, molecule2, label = datapoint
            molecule1, molecule2 = molecule1.to(device), molecule2.to(device)
            label = label.to(device).unsqueeze(-1).float()
            
            optimizer.zero_grad()
            output = model(molecule1, molecule2)
            
            batch_loss = loss_fn(output, label)
            total_loss += batch_loss.item()
            num_batches += 1
            
            pred_interaction = (output > 0.5)
            correct_preds += (pred_interaction == label).sum().item()
            total_preds += label.size(0)
            
            batch_loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({
                'loss': f"{batch_loss.item():.4f}",
                'accuracy': f"{correct_preds/total_preds:.4f}"
            })
            
        
        epoch_loss = total_loss / num_batches
        epoch_accuracy = correct_preds / total_preds
        
        history['epoch_losses'].append(epoch_loss)
        history['epoch_accuracies'].append(epoch_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Accuracy = {epoch_accuracy:.4f}, Loss = {epoch_loss:.4f}")
      
    if save_model:
        save_model_and_history(model, history, save_dir)

    print(history)

    if plot_training_curve:
        plot_training_curves(history)

    return model, history

if __name__ == '__main__':
    # Load data
    tsv_file = 'processed_data/combined_drug_pairs.tsv.gz'
    train_loader, test_loader = create_data_loaders(
        tsv_file,
        batch_size=32,
        cache_dir='./data_cache'
    )

    # Initialize model

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

    model = LSTM_DDI(
        gnn = gnn,
        embed_dim = 128,
        hidden_dim = 256,
        num_layers = 4,
        activation_fn = torch.nn.GELU()
    )

    # Training
    trained_model, training_history = train(
        model=model,
        train_loader=train_loader,
        num_epochs=10,
        plot_training_curve=False
    )

    # Evaluation
    eval_metrics = evaluate_model(
        model=model,
        test_loader=test_loader
    )
