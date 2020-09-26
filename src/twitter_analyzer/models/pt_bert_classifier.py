import os
import torch
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from transformers import *
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


class PTBertClassifier:

    def __init__(self,
                 num_classes,
                 tokenizer=BertTokenizer.from_pretrained("bert-base-uncased"),
                 transf_model=BertForSequenceClassification.from_pretrained("bert-base-uncased")
                 ):
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.model = transf_model

    def get_features(self,
                     x,
                     y):
        processor = SingleSentenceClassificationProcessor()
        processor.add_examples(texts_or_text_and_labels=x,
                               labels=y)

        features = processor.get_features(tokenizer=self.tokenizer,
                                          return_tensors="pt")
        return features

    def fit(self,
            x,
            y,
            epochs=10,
            lr=3e-5,
            batch_size=8,
            val_split=0.7,
            model_save_path=None,
            ):

        x_train, x_valid, y_train, y_valid = train_test_split(x,
                                                              y,
                                                              train_size=val_split)

        train_features = self.get_features(x=x_train, y=y_train)
        valid_features = self.get_features(x=x_valid, y=y_valid)

        train_input_ids = torch.tensor(np.array(train_features[:][0]))
        train_input_mask = torch.tensor(np.array(train_features[:][1]))
        train_label = torch.tensor(np.array(train_features[:][2]))

        valid_input_ids = torch.tensor(np.array(valid_features[:][0]))
        valid_input_mask = torch.tensor(np.array(valid_features[:][1]))
        valid_label = torch.tensor(np.array(valid_features[:][2]))

        train = torch.utils.data.TensorDataset(train_input_ids, train_input_mask, train_label)
        valid = torch.utils.data.TensorDataset(valid_input_ids, valid_input_mask, valid_label)

        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(),
                               lr=lr)

        self.model.cuda()
        self.model.train()

        for e in range(epochs):
            train_loss = 0.0

            for batch in tqdm(train_loader, position=0, leave=True):
                batch = tuple(t.cuda() for t in batch)

                x_ids, x_masks, y_truth = batch

                y_pred = self.model(x_ids, x_masks)
                loss = loss_fn(y_pred[0], y_truth)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() / len(train_loader)

            self.model.eval()
            val_loss = 0
            truths = None
            preds = None
            with torch.no_grad():
                for i, batch in tqdm(enumerate(valid_loader), position=0, leave=True):
                    batch = tuple(t.cuda() for t in batch)
                    x_ids, x_masks, y_truth = batch

                    logits = self.model(x_ids, x_masks)[0].detach()

                    val_loss += loss_fn(logits, y_truth).item() / len(valid_loader)

                    if preds is None:
                        preds = logits.detach().cpu().numpy()
                        truths = y_truth.detach().cpu().numpy()
                    else:
                        preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                        truths = np.append(truths, y_truth.detach().cpu().numpy(), axis=0)

            preds = np.argmax(preds, axis=1)
            acc, f1 = metric(preds, truths)

            print(
                'epoch: %d, train loss: %.8f, valid loss: %.8f, acc: %.8f, f1: %.8f\n' %
                (e, train_loss, val_loss, acc, f1))

            torch.cuda.empty_cache()

        if model_save_path is not None:
            torch.save(self.model, os.path.join(model_save_path, "model.bin"))

    def predict(self, text):
        feature = self.tokenizer.encode_plus(text=text, add_special_tokens=True, return_tensors='pt', max_length=512)
        prediction = self.model(feature['input_ids'], feature['attention_mask'])[
            0].detach().cpu()
        return F.softmax(prediction)

    def predict_sentiment(self, text, thresh=0.6):
        feature = self.tokenizer.encode_plus(text=text, add_special_tokens=True, return_tensors='pt', max_length=512)
        prediction = self.model(feature['input_ids'], feature['attention_mask'])[
            0].detach().cpu()
        preds = F.softmax(prediction)

        pred = (preds > thresh).byte().numpy()[0]

        if all(pred == np.array([0, 0])):
            sentiment = "neutral"
        elif all(pred == np.array([1, 0])):
            sentiment = "positive"
        else:
            sentiment = "negative"

        conf = preds.max().item()

        return sentiment, conf