import numpy as np
import matplotlib.pyplot as plt
import torch
import random
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    roc_auc_score, 
    average_precision_score, 
    classification_report,
    precision_recall_curve,
    auc
)

# Use GPU training if it is availabe
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Ealuation Metrics
def calculate_metric(model,
                     data_loader,
                     metric="auprc",
                     true_labels=None,
                     pred_labels=None):
    """ Function to evaluate model performance on a given metric
      Args:
        model: Pytorch model
        data_loader: PyTorch dataloader (test/validation)
        metric: Metric to evaluate, e.g., "auprc", "acc", "f1", "precision", "recall", "roc_auc", "cls_report"

      Returns:
        metric_evaluated: float or report based on the chosen metric
    """
    if true_labels is None and pred_labels is None:
        with torch.no_grad():
            pred_labels, true_labels = [], []
            for features, labels in data_loader:
                # features: tensor shape [bs, feature_size], labels: tensor shape [bs, 1]
                features, labels = features.to(device), labels.to(device)
                output = model(features) # output: tensor shape [bs, 1]

                predictions = output.squeeze().cpu().numpy()  # Use raw output for probability-based metrics
                pred_labels.extend(predictions.tolist())
                true_labels.extend(labels.squeeze(-1).tolist())

    if metric == "auprc":
        precision, recall, _ = precision_recall_curve(true_labels, pred_labels)
        metric_evaluated = auc(recall, precision)
    elif metric == "acc":
        metric_fc = accuracy_score
    elif metric == "f1":
        metric_fc = f1_score
    elif metric == "precision":
        metric_fc = precision_score
    elif metric == "recall":
        metric_fc = recall_score
    elif metric == "roc_auc":
        metric_fc = roc_auc_score
    elif metric == "cls_report":
        return classification_report(true_labels, (np.array(pred_labels) > 0.5).astype(int), target_names=["Fraudulent", "Non-Fraudulent"])
    else:
        metric_fc = accuracy_score

    # For metrics that aren't AUPRC, compute directly using thresholded labels
    if metric != "auprc":
        metric_evaluated = metric_fc(true_labels, (np.array(pred_labels) > 0.5).astype(int))

    return metric_evaluated


def train(model,
          train_loader,
          val_loader,
          num_epochs=50,
          optimizer=None,
          criterion=None,
          seed=53,
          exp_name=None,
          metric="auprc"):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_loss_history = [] # store the loss
    train_auprc_history = [] # store the AUPRC
    valid_auprc_history = [] # store the AUPRC
    best_val_auprc = 0  # To track the best AUPRC

    model = model.to(device)
    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Training Loop
        model.train()
        train_loss = 0
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output.squeeze(), labels)
            if (i+1) % 5 == 0:
                train_loss_history.append(loss.item())
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        # Evaluation Loop
        model.eval()
        train_auprc = calculate_metric(model, train_loader, metric)
        valid_auprc = calculate_metric(model, val_loader, metric)
        print(f"Epoch {epoch} | Training Loss: {train_loss/len(train_loader):.4f}  {metric}: trainset={train_auprc:.4f} valset={valid_auprc:.4f}")

        # Save the best model based on validation AUPRC
        if valid_auprc > best_val_auprc:
            best_val_auprc = valid_auprc
            torch.save(model.state_dict(), f"./{exp_name}_best.pt")
            print(f"Saving Best Model at Epoch {epoch}")

        train_auprc_history.append(train_auprc)
        valid_auprc_history.append(valid_auprc)

    model.load_state_dict(torch.load(f"./{exp_name}_best.pt", map_location=device))  # Load the best model for further inference

    return train_loss_history, train_auprc_history, valid_auprc_history, model

# Evalution Visualization
def evaluate_model(exp_name,
                   model,
                   train_loss_history,
                   train_auprc_history,
                   valid_auprc_history,
                   test_loader,
                   metric="auprc"):

    print(f"Best Training {metric.upper()}: {np.max(train_auprc_history):.4f}")
    print(f"Best Validation {metric.upper()}: {np.max(valid_auprc_history):.4f}")

    # Create a figure with two subplots
    fig, ax = plt.subplots(figsize=(12, 6), ncols=2)

    # Plot for Training Loss
    ax[0].plot(range(1, len(train_loss_history)+1), train_loss_history, label='Training Loss')
    ax[0].set_xlabel('Steps')
    ax[0].set_ylabel('Loss')
    ax[0].set_title(f'{exp_name} Training Loss')
    ax[0].grid(True)
    ax[0].legend()

    # Plot for AUPRC
    ax[1].plot(range(1, len(train_auprc_history)+1), train_auprc_history, label=f'Training {metric.upper()}', color='blue')
    ax[1].plot(range(1, len(valid_auprc_history)+1), valid_auprc_history, label=f'Validation {metric.upper()}', color='green')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel(metric.upper())
    ax[1].set_title(f'{exp_name} Performance Over Epochs')
    ax[1].legend()
    ax[1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plot
    plt.show()

    # Evaluate and print final performance on the test set
    test_set_performance = calculate_metric(model, test_loader, metric="cls_report")
    print("Test Set Performance:")
    print(test_set_performance)


