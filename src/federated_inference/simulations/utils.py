
import math
import random
import os
import json

import torch
import numpy as np
import pandas as pd

import plotly.graph_objects as go
import plotly.subplots as sp
from IPython.display import display

from federated_inference.dataset.client import ClientDataset
from federated_inference.transform.data_splitter import DataSplitter


def tensor_to_numpy_image(tensor_img):
    if not isinstance(tensor_img, torch.Tensor):
        return tensor_img

    img = tensor_img.detach().cpu().numpy()

    if img.ndim == 3:
        if img.shape[0] == 1:
            img = img.squeeze(0)
        else:
            img = np.transpose(img, (1, 2, 0))

    if np.issubdtype(img.dtype, np.floating):
        min_val, max_val = img.min(), img.max()
        if max_val <= 1.0 and min_val >= 0.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = (255 * (img - min_val) / (max_val - min_val + 1e-5)).astype(np.uint8)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    return img


def create_simulation_image_subplots(simulation, idx=0):
    datasets = simulation.client_datasets
    if isinstance(datasets[0], ClientDataset):
        datasets = [d.train_dataset for d in datasets]

    n_clients = len(datasets)
    cols = math.floor(n_clients / 2) if n_clients > 3 else 1
    rows = math.ceil(n_clients / cols)

    fig = sp.make_subplots(rows=rows, cols=cols,
                           subplot_titles=[f"Client {i + 1}" for i in range(n_clients)],
                           vertical_spacing=0.08,
                           horizontal_spacing=0.01)

    for i, dataset in enumerate(datasets):
        r, c = divmod(i, cols)
        img, _ = dataset[idx]
        img_np = tensor_to_numpy_image(img)
        fig.add_trace(go.Image(z=img_np), row=r + 1, col=c + 1)

    fig.update_layout(height=250 * rows, width=250 * cols, showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig


def create_client_image_subplots(simulation, client_indices=[0], n_images=3, keys=None):
    datasets = simulation.client_datasets
    if isinstance(datasets[0], ClientDataset):
        datasets = [d.train_dataset for d in datasets]

    grouped_trainset = DataSplitter.group_dataset(simulation.clients[0].data.train_dataset)
    grouped_trainset = {k: grouped_trainset[k] for k in sorted(grouped_trainset)}

    k_indices = {}
    n_clients, n_keys = len(client_indices), len(keys)
    rows, cols = n_clients * n_keys, n_images

    fig = sp.make_subplots(rows=rows, cols=cols,
                           vertical_spacing=0.2,
                           horizontal_spacing=0.01)

    for k, key in enumerate(keys):
        indices = grouped_trainset.get(key, [])
        if len(indices) < n_images:
            continue

        sampled_idx = random.sample(indices, n_images)
        k_indices[key] = sampled_idx

        for r, client_idx in enumerate(client_indices):
            dataset = datasets[client_idx]
            row = k * n_clients + r + 1
            for c in range(n_images):
                img, _ = dataset[sampled_idx[c]]
                img_np = tensor_to_numpy_image(img)
                fig.add_trace(go.Image(z=img_np), row=row, col=c + 1)

            fig.update_yaxes(title_text=f"Client {client_idx} <br> Label {key}", row=row, col=1)

    fig.update_layout(
        height=rows * 130,
        width=cols * 130,
        showlegend=False,
        margin=dict(t=30, b=30)
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig, k_indices


def show_simulation_data(simulation):
    fig, _ = create_client_image_subplots(simulation, [0], 8, keys=[8, 9])
    display(fig)


def plot_loss_curve(losses, log_interval, idx, _type="Training", x_label="Batch"):
    x_values = [i * log_interval for i in range(len(losses))]

    fig = go.Figure(go.Scatter(
        x=x_values,
        y=losses,
        mode='lines+markers',
        name=f"{_type} Loss",
        line=dict(color='royalblue'),
        marker=dict(size=4)
    ))

    fig.update_layout(
        title=f"Isolated MNIST {_type} Loss Over Time of Client {idx}",
        xaxis_title=x_label,
        yaxis_title='Loss',
        template='plotly_white',
        height=400,
        width=700
    )
    return fig


def plot_training_loss(train_losses, log_interval, idx, _type="Training"):
    return plot_loss_curve(train_losses, log_interval, idx, _type, x_label="Batch")


def plot_test_loss(test_losses, log_interval, idx, _type="Test"):
    return plot_loss_curve(test_losses, log_interval, idx, _type, x_label="Epoch")


def print_cm_heat(cm_df, idx):
    fig = go.Figure(data=go.Heatmap(
        z=cm_df.values,
        x=cm_df.columns,
        y=cm_df.index,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        hoverongaps=False,
        colorbar=dict(title="Proportion")
    ))

    fig.update_layout(
        title=f"MNIST Normalized Confusion Matrix of Client {idx}",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        xaxis=dict(tickangle=45),
        height=600,
        width=700
    )

    return fig


def cm_analysis(client):
    cm = client.cm.to_numpy()
    labels = client.labels
    numerical_labels = range(len(labels))

    cm_percent = cm.astype(np.float32)
    cm_percent = cm_percent / cm_percent.sum(axis=1, keepdims=True)
    df_cm_per = pd.DataFrame(cm_percent, index=[f'True {l}' for l in labels],
                             columns=[f'Pred {l}' for l in labels])

    TP = np.diag(cm)
    precision = TP / np.sum(cm, axis=0)
    recall = TP / np.sum(cm, axis=1)
    correct = np.diag(cm_percent)
    correct_count = np.diag(cm)

    accuracy = np.sum(correct_count) / np.sum(cm)

    analysis = {
        "correct": {
            "most_correct_class": int(np.argmax(correct)),
            "max_correct_value": float(np.max(correct))
        },
        "performance": {
            "accuracy": float(accuracy),
            "precision": [float(p) for p in precision],
            "recall": [float(r) for r in recall]
        }
    }

    # Incorrect predictions
    cm_no_diag = cm_percent.copy()
    np.fill_diagonal(cm_no_diag, 0)
    max_wrong_idx = np.unravel_index(np.argmax(cm_no_diag), cm_no_diag.shape)
    wrong_from, wrong_to = max_wrong_idx

    wrong_per_class = cm_no_diag.sum(axis=1)
    most_misclassified_class = int(np.argmax(wrong_per_class))
    most_misclassified_count = float(np.max(wrong_per_class))

    analysis["wrong"] = {
        "max_wrong_idx": [int(i) for i in max_wrong_idx],
        "max_wrong_count": float(cm_no_diag[max_wrong_idx]),
        "wrong_from": int(wrong_from),
        "wrong_to": int(wrong_to),
        "most_misclassified_class": most_misclassified_class,
        "most_misclassified_count": most_misclassified_count
    }

    print(f"\nMost correct class: {analysis['correct']['most_correct_class']} "
          f"({analysis['correct']['max_correct_value']} correct predictions)")

    print(f"Most wrong prediction: class {wrong_from} misclassified as {wrong_to} "
          f"({analysis['wrong']['max_wrong_count']} times)")

    print(f"Most misclassified class: {most_misclassified_class} "
          f"({most_misclassified_count} samples misclassified in total)")

    return analysis, df_cm_per




