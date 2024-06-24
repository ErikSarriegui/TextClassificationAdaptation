from datasets import DatasetDict, Dataset
import pandas as pd
import numpy as np

def preprocessDataset(examples, classification_labels, tokenizer):
    """
    Esta función toma un conjunto de datos, lo tokeniza y añade las etiquetas de clasificación al
    apartado "labels" junto al texto tokenizado.

    Parámetros:
    - examples: diccionario que contiene los datos, incluido el texto bajo la clave "post_text".
    - classification_labels: lista de nombres de las etiquetas de clasificación (por ejemplo, ["symptom0", "symptom1", ...]).
    - tokenizer: objeto tokenizer que se usa para tokenizar el texto.

    Output:
    - encoding: diccionario con los datos tokenizados y las etiquetas de clasificación añadidas bajo la clave "labels".
    """
    text_data = examples["text"]
    encoding = tokenizer(text_data, padding="max_length", truncation=True, max_length=128)

    # Filtrar las etiquetas de clasificación del dataset
    filtered_labels = {label: examples[label] for label in classification_labels}

    # Crear una matriz de etiquetas con ceros
    labels_matrix = np.zeros((len(text_data), len(classification_labels)))

    # Rellenar la matriz de etiquetas
    for idx, label in enumerate(classification_labels):
        labels_matrix[:, idx] = filtered_labels[label]

    # Añadir la matriz de etiquetas al diccionario de encoding
    encoding["labels"] = labels_matrix.tolist()

    return encoding
