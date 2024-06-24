from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
import torch

class TrainLLM:
    @staticmethod
    def compute_multi_label_metrics(predictions, true_labels, threshold=0.5):
        """
        Calcula métricas para clasificación multi-etiqueta.

        Parámetros:
        - predictions: array de predicciones del modelo.
        - true_labels: array de etiquetas reales.
        - threshold: umbral para convertir probabilidades en predicciones binarias.

        Output:
        - Diccionario con métricas f1, roc_auc y accuracy.
        """
        softmax = torch.nn.Softmax(dim=1)
        probabilities = softmax(torch.Tensor(predictions))
        predicted_classes = np.argmax(probabilities.numpy(), axis=1)

        f1_micro_avg = f1_score(y_true=true_labels, y_pred=predicted_classes, average='micro')
        roc_auc = roc_auc_score(y_true=true_labels, y_score=probabilities.numpy(), multi_class='ovr', average='macro')
        accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_classes)

        metrics = {
            'f1': f1_micro_avg,
            'roc_auc': roc_auc,
            'accuracy': accuracy
        }
        return metrics

    def compute_metrics(self, evaluation_prediction: EvalPrediction):
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
        model,
        train_dataset,
        eval_dataset,
        tokenizer,
        batch_size = 8,
        metric_name = "f1",
        learning_rate = "2e-5",
        num_train_epochs = 5,
        weight_decay = 0.01
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
        training_args = TrainingArguments(
            output_dir="/finetuned_model",
            evaluation_strategy="epoch",
            save_strategy="epoch",
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
            compute_metrics=self.compute_metrics
        )

        trainer.train()
        return model
