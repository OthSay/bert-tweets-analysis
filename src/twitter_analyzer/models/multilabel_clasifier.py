from sklearn.model_selection import train_test_split
import torch
from transformers import *
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor

from twitter_analyzer.data import MultiLabelClassificationProcessor


def accuracy_thresh(y_pred, y_true, thresh: float = 0.5, sigmoid: bool = True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    if sigmoid: y_pred = y_pred.sigmoid()
    return np.mean(np.mean(((y_pred > thresh) == y_true.byte()).float().cpu().numpy(), axis=1))


label_map = {0: 'toxic',
             1: 'severe_toxic',
             2: 'obscene',
             3: 'threat',
             4: 'insult',
             5: 'identity_hate'}


class MultiLabelClassifier:

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
        processor = MultiLabelClassificationProcessor(mode="multilabel_classification")
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
            model_save_path="weights_imdb.{epoch:02d}.hdf5"
            ):

        x_train, x_valid, y_train, y_valid = train_test_split(x,
                                                              y,
                                                              train_size=val_split)

        train_features = self.get_features(x=x_train, y=y_train)
        valid_features = self.get_features(x=x_valid, y=y_valid)

        train_input_ids = tensor(np.array(train_features[:][0]))
        train_input_mask = tensor(np.array(train_features[:][1]))
        train_label = tensor(np.array(train_features[:][2]))

        valid_input_ids = tensor(np.array(valid_features[:][0]))
        valid_input_mask = tensor(np.array(valid_features[:][1]))
        valid_label = tensor(np.array(valid_features[:][2]))

        train = TensorDataset(train_input_ids, train_input_mask, train_label)
        valid = TensorDataset(valid_input_ids, valid_input_mask, valid_label)

        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

        loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model.cuda()
        optimizer = optim.Adam(params=self.model.parameters(), lr=lr)
        self.model.train()

        self.model.cuda()
        self.model.train()

        for e in range(epochs):
            train_loss = 0.0

            for batch in tqdm(train_loader, position=0, leave=True):
                batch = tuple(t.cuda() for t in batch)

                x_ids, x_masks, y_truth = batch

                y_pred = self.model(x_ids, x_masks)
                logits = y_pred[0]
                loss = loss_fn(logits, y_truth)
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

            acc = accuracy_thresh(tensor(preds), tensor(truths))

            print(
                'epoch: %d, train loss: %.8f, valid loss: %.8f, acc: %.8f\n' %
                (e, train_loss, val_loss, acc[0]))

            torch.cuda.empty_cache()

        torch.save(self.model, model_save_path)

    def predict(self, text, thresh=0.5):

        feature = self.tokenizer.encode_plus(text=text, add_special_tokens=True, return_tensors='pt', max_length=512)
        prediction = self.model(feature['input_ids'], feature['attention_mask'])[0].detach().cpu().sigmoid()

        preds = (prediction > thresh).byte().numpy()
        pred_idx = np.where(preds[0] == 1)[0]

        return [label_map[idx] for idx in pred_idx]
