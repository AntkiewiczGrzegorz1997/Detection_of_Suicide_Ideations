from transformers import BertTokenizer
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)



dataset = "aladag"
if dataset == "reddit_500":


    df_mental = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/reddit_500_final_train.csv")
#df_mental = df_mental.astype({"text": str, "label":str}, errors='raise')
    df_mental = df_mental.rename(columns={"Label": "label"})



    df_mental_valid = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/reddit_500_final_val.csv")
    df_mental_valid = df_mental_valid.rename(columns={"Label": "label"})

#df_mental_valid = df_mental_valid.astype({"text": str, "label":str}, errors='raise')
elif dataset == "aladag":
    #add path to aladag
    df_mental = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/Aladag_sample_preprocessed.csv")
    df_mental_valid = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/Aladag_labeled_preprocessed.csv")


    df_mental = df_mental.rename(columns={"binary_annotation": "label"})
    df_mental_valid = df_mental_valid.rename(columns={"binary_annotation": "label"})
    df_mental_valid.label = df_mental_valid.label.astype(int)
elif dataset== "smhd":

    df_mental = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/df_mental_balanced_preprocessed_preprocessed.csv")
    df_mental_valid = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/df_mental_valid_preprocessed_preprocessed.csv")
    #df_mental = df_mental.rename(columns={"binary_annotation": "label"})
    #df_mental_valid = df_mental_valid.rename(columns={"binary_annotation": "label"})
    #df_mental_valid.label = df_mental_valid.label.astype(int)
    #df_mental.label = df_mental.label.astype(int)
    df_mental_valid.selftext = df_mental_valid.text
    df_mental.selftext = df_mental.text
    df_mental = df_mental.astype({"selftext": str, "label":str}, errors='raise')
    df_mental_valid = df_mental.astype({"selftext": str, "label":str}, errors='raise')

    possible_labels = sorted(df_mental.label.unique())

    label_dict = {}
    for index, possible_label in enumerate(possible_labels):
        label_dict[possible_label] = index

    df_mental['label'] = df_mental.label.replace(label_dict)
    df_mental_valid['label'] = df_mental_valid.label.replace(label_dict)

elif dataset=="rsdd":

    df_mental = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/rsdd_train_preprocessed.csv")
    df_mental_valid = pd.read_csv("/Users/grzegorzantkiewicz/Downloads/rsdd_valid_preprocessed.csv")

    df_mental_valid.selftext = df_mental_valid.posts
    df_mental.selftext = df_mental.posts
    df_mental_valid.label = df_mental_valid.label.astype(int)
    df_mental.label = df_mental.label.astype(int)





#print(df_mental["text"])
#print(df_mental_valid.dtypes)


encoded_data_train = tokenizer.batch_encode_plus(
    df_mental.selftext.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    truncation = True,
    return_tensors='pt'
)

encoded_data_val= tokenizer.batch_encode_plus(
    df_mental_valid.selftext.values,
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    truncation=True,
    return_tensors='pt'
)

possible_labels = sorted(df_mental.label.unique())

label_dict = {}
for index, possible_label in enumerate(possible_labels):
    label_dict[possible_label] = index

#df_mental['code'] = df_mental.label.replace(label_dict)
#df_mental_valid['code'] = df_mental_valid.label.replace(label_dict)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(df_mental.label.values)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df_mental_valid.label.values)

dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 6

dataloader_train = DataLoader(dataset_train,
                              sampler=RandomSampler(df_mental),
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val
                                   ,
                                   sampler=SequentialSampler(df_mental_valid),
                                   batch_size=batch_size)

from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(),
                  lr=1e-5,
                  eps=1e-8)

epochs = 3

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train) * epochs)




def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def accuracy_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, preds_flat)

def precision_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return precision_score(labels_flat, preds_flat, average='weighted')

def recall_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return recall_score(labels_flat, preds_flat, average='weighted')





def accuracy_per_class(preds, labels):
    label_dict_inverse = {v: k for k, v in label_dict.items()}

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()

    for label in np.unique(labels_flat):
        y_preds = preds_flat[labels_flat == label]
        y_true = labels_flat[labels_flat == label]
        print(f'Class: {label_dict_inverse[label]}')
        print(f'Accuracy: {len(y_preds[y_preds == label])}/{len(y_true)}\n')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(device)

from tqdm.notebook import tqdm
import numpy as np


def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


import random

seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def evaluate(dataloader_val):
    model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []

    for batch in dataloader_val:
        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        with torch.no_grad():
            outputs = model(**inputs)

        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)

    loss_val_avg = loss_val_total / len(dataloader_val)

    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)

    return loss_val_avg, predictions, true_vals


for epoch in tqdm(range(1, epochs + 1)):

    model.train()

    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:
        model.zero_grad()

        batch = tuple(b.to(device) for b in batch)

        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2],
                  }

        outputs = model(**inputs)

        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item() / len(batch))})

    torch.save(model.state_dict(), f'finetuned_BERT_2classes_aladag_LASTversion{epoch}.model')

    tqdm.write(f'\nEpoch {epoch}')

    loss_train_avg = loss_train_total / len(dataloader_train)
    tqdm.write(f'Training loss: {loss_train_avg}')

    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_score_func(predictions, true_vals)
    val_accuracy = accuracy_score_func(predictions, true_vals)
    val_prec = precision_score_func(predictions, true_vals)
    val_rec = recall_score_func(predictions, true_vals)

    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score: {val_f1}')
    tqdm.write(f'Accuracy: {val_accuracy}')
    tqdm.write(f'Precision: {val_prec}')
    tqdm.write(f'Recall: {val_rec}')
