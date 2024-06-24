from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, DatasetDict
import pandas as pd
from trainer import TextClassificationAdaptation
from utils import preprocessDataset

"""
CARGAMOS LOS DATOS

- Tenemos que tener un DataFrame que tenga una columna "text" que sea el texto y todas las demás columnas serán labels
- Hay que tener cuidado porque, en este caso, únicamente estamos cargando un dataset (train), idealmente hay que tener
dos, uno de entrenamiento y otro de validación
"""

data = [
    ["Esto es un texto de prueba", False, True, False],
    ["Esto es otro texto de prueba", False, False, True]
]

df = pd.DataFrame(data, columns=["text", "symptom0", "symptom1", "symptom2"])
dataset = DatasetDict({"train" : Dataset.from_pandas(df)})

"""
EXTRACCIÓN DE LABELS
"""
labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}


"""
CARGAR MODELO PRE-ENTRENADO A ADAPTAR
"""
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, problem_type="multi_label_classification", num_labels=len(labels), id2label=id2label, label2id=label2id)


"""
PREPROCESADO DEL DATASET
"""
encoded_dataset = dataset.map(
    preprocessDataset,
    batched=True,
    remove_columns=dataset['train'].column_names,
    fn_kwargs={'classification_labels': labels, 'tokenizer': tokenizer}
)

encoded_dataset.set_format("torch")


"""
UTILIZACIÓN DE TextClassificationAdaptation para realizar la adaptación
"""
trainer = TextClassificationAdaptation()
finetuned_model = trainer.train(
    model = model,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["train"],
    tokenizer = tokenizer
)

"""
GUARDAR MODELO Y TOKENIZADOR
"""
finetuned_model.save_pretrained("TextClassification/model")
tokenizer.save_pretrained("TextClassification/tokenizer")
