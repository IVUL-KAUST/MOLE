from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
from sklearn.metrics import confusion_matrix
id2label = {0: 'ar', 1: 'en', 2: 'fr', 3: 'jp', 4: 'ru', 5: 'multi', 6: 'other'}
label2id = {v: k for k, v in id2label.items()}

dataset = load_dataset('csv', data_files='train_dataset.csv', split='train').train_test_split(test_size=0.1, shuffle=True)
train_dataset = dataset['train']
validation_dataset = dataset['test']
test_dataset = load_dataset('csv', data_files='test_dataset.csv', split='train')
print(train_dataset)
print(validation_dataset)
print(test_dataset)

model_name = 'bert-base-uncased'
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 7, id2label=id2label, label2id=label2id)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def process_data(examples):
    examples['label'] = label2id[examples['schema_name']]
    if examples['abstract'] is not None:
        examples['text'] = examples['title'] + ' ' + examples['abstract']
    else:
        examples['text'] = examples['title']
    return examples

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)


train_dataset = train_dataset.map(process_data)
validation_dataset = validation_dataset.map(process_data)
test_dataset = test_dataset.map(process_data)

train_dataset = train_dataset.map(tokenize_function)
validation_dataset = validation_dataset.map(tokenize_function)
test_dataset = test_dataset.map(tokenize_function)


training_args = TrainingArguments(
    output_dir='./results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=7,
    eval_strategy="epoch",
    logging_steps=1,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    top_2_accuracy = top_k_accuracy_score(labels, logits, k=2, labels= list(range(7)))
    return {
        'accuracy': accuracy_score(predictions, labels),
        'f1': f1_score(predictions, labels, average='macro'),
        'top_2_accuracy': top_2_accuracy
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print(trainer.evaluate(test_dataset))

# confusion matrix
pred_labels = trainer.predict(test_dataset)
pred_labels = list(np.argmax(pred_labels.predictions, axis=-1))
gold_labels = list(test_dataset['label'])
# plot confusion matrix
print(confusion_matrix(gold_labels, pred_labels))




