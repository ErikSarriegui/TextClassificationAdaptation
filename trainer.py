from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from datasets import Dataset, DatasetDict
from typing import List, Dict
import pandas as pd
import numpy as np
import torch

class TextClassificationAdaptation:
    """
    Código de: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
    """
    def __init__(self) -> None:
        self.results = []

    def compute_multi_label_metrics(
        self,
        predictions : List,
        true_labels : List,
        threshold : int = 0.5
    ) -> Dict:
        """
        Calcula métricas para clasificación multi-etiqueta. Hay que tener cuidado porque habitualmente la función sigmoidal se utiliza
        para tareas de clasificación binaria, pero, como en el tutorial está, no he querido cambiarlo.

        Parámetros:
        - predictions: array de predicciones del modelo.
        - true_labels: array de etiquetas reales.
        - threshold: umbral para convertir probabilidades en predicciones binarias.

        Output:
        - Diccionario con métricas f1, roc_auc y accuracy.
        """
        probabilities = torch.nn.functional.softmax(torch.tensor(predictions), dim=-1).numpy()
        predicted_classes = np.argmax(probabilities, axis=1)
    
        f1 = f1_score(y_true=true_labels, y_pred=predicted_classes, average='micro')
        accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_classes)
        roc_auc = roc_auc_score(y_true=true_labels, y_score=probabilities, multi_class='ovr')

        result_dict = {
            'f1': f1_micro_avg,
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }
        self.results.append(result_dict)
        return result_dict

    def compute_metrics(
        self,
        evaluation_prediction: EvalPrediction
    ) -> Dict:
        """
        Computa las métricas requeridas durante la evaluación del modelo.

        Parámetros:
        - evaluation_prediction: objeto EvalPrediction que contiene predicciones y etiquetas reales.

        Output:
        - Diccionario con las métricas calculadas.
        """
        predictions = evaluation_prediction.predictions[0] if isinstance(evaluation_prediction.predictions, tuple) else evaluation_prediction.predictions
        return self.compute_multi_label_metrics(predictions=predictions, true_labels=evaluation_prediction.label_ids)

    def train(
        self,
        model: AutoModelForSequenceClassification,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: AutoTokenizer,
        batch_size: int = 8,
        metric_name: str = "f1",
        learning_rate: float = 2e-5,
        num_train_epochs: int = 5,
        weight_decay: float = 0.01,
        optimizer = None,
        lr_scheduler = None,
        eval_strategy = "epoch"
    ):
        """
        Entrena el modelo utilizando los conjuntos de datos de entrenamiento y evaluación proporcionados.

        Parámetros:
        - model: modelo a ser entrenado.
        - train_dataset: conjunto de datos de entrenamiento.
        - eval_dataset: conjunto de datos de evaluación.
        - tokenizer: tokenizer a ser utilizado.
        - batch_size: tamaño del batch para entrenamiento y evaluación.
        - metric_name: métrica principal para seleccionar el mejor modelo.

        Output:
        - Modelo entrenado.
        """
        total_steps = len(train_dataset) // batch_size * num_train_epochs

        def lr_lambda(current_step: int):
            return max(0.0, float(1 - current_step / total_steps))
        
        if lr_scheduler is None:
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if optimizer is None:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        training_args = TrainingArguments(
            output_dir="/finetuned_model",
            eval_strategy=eval_strategy,
            save_strategy=eval_strategy,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=metric_name,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=self.compute_metrics,
            optimizers=(optimizer, lr_scheduler)
        )

        trainer.train()

        return model, {
            "epoch" : list(range(1, len(self.results) + 1)),
            "f1" : [element["f1"] for element in self.results],
            "roc_auc" : [element["roc_auc"] for element in self.results],
            "accuracy" : [element["accuracy"] for element in self.results],
        }
