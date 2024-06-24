import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trainer import TrainLLM
from utils import preprocessDataset

data = [
    ["Esto es un texto de prueba", False, True, False],
    ["Esto es otro texto de prueba", False, False, False]
]

df = pd.DataFrame(data, columns=["text", "symptom0", "symptom1", "symptom2"])

# Prepare data 2 dataset
dataset = DatasetDict({"train" : Dataset.from_pandas(df)})

# Labels
labels = [label for label in dataset['train'].features.keys() if label not in ['text']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", problem_type="multi_label_classification", num_labels=len(labels), id2label=id2label, label2id=label2id)

# Tokenize & Preprocess data
encoded_dataset = dataset.map(preprocessDataset, batched=True, remove_columns=dataset['train'].column_names, fn_kwargs={'classification_labels': labels, 'tokenizer': tokenizer})
encoded_dataset.set_format("torch")

trainer = TrainLLM()

finetuned_model = trainer.train(
    model = model,
    train_dataset = encoded_dataset["train"],
    eval_dataset = encoded_dataset["train"],
    tokenizer = tokenizer
)

# Para guardarlo
finetuned_model.save_pretrained("TextClassification/model")
tokenizer.save_pretrained("TextClassification/tokenizer")
