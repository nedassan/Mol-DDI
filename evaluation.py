import pandas as pd
import numpy as np
import torch
import gzip
import csv
from tqdm import tqdm

def evaluate_model(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.eval()
    
    all_sigmoid_values = []
    all_predictions = []
    all_labels = []
    total_loss = 0
    num_batches = 0
    
    loss_fn = torch.nn.BCELoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            molecule1, molecule2, labels = batch
            molecule1, molecule2 = molecule1.to(device), molecule2.to(device)
            labels = labels.to(device).unsqueeze(-1).float()
            
            outputs = model(molecule1, molecule2)
            batch_loss = loss_fn(outputs, labels)
            total_loss += batch_loss.item()
            num_batches += 1
            
            sigmoid_values = outputs.cpu().numpy()
            predictions = (sigmoid_values > 0.5).astype(int)
            
            all_sigmoid_values.extend(sigmoid_values.flatten())
            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    avg_loss = total_loss / num_batches
    
    metrics = {
        'accuracy': accuracy,
        'loss': avg_loss,
        'num_samples': len(all_labels)
    }
    
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Test Loss:       {avg_loss:.4f}")
    print(f"Test Accuracy:   {accuracy*100:.2f}%")
    print(f"Number of pairs: {len(all_labels)}")
    print(f"True Positives:  {np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 1))}")
    print(f"True Negatives:  {np.sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 0))}")
    print(f"False Positives: {np.sum((np.array(all_predictions) == 1) & (np.array(all_labels) == 0))}")
    print(f"False Negatives: {np.sum((np.array(all_predictions) == 0) & (np.array(all_labels) == 1))}")
    print("="*50)

    
    return metrics

    
