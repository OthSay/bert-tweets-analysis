import tensorflow as tf
from transformers.modeling_tf_bert import TFBertForSequenceClassification
from transformers.data.processors.utils import SingleSentenceClassificationProcessor
from transformers.tokenization_bert import BertTokenizer


class TFBertClassifier:
    """
        Super Model Class
    """

    def __init__(self,
                 num_classes,
                 model_name='bert-base-uncased'
                 ):
        self.num_classes = num_classes
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = TFBertForSequenceClassification.from_pretrained(model_name,
                                                                     num_labels=self.num_classes)

    def get_features(self,
                     x,
                     y,
                     batch_size=8,
                     shuffle=True):
        processor = SingleSentenceClassificationProcessor()
        processor.add_examples(texts_or_text_and_labels=x,
                               labels=y)

        features = processor.get_features(tokenizer=self.tokenizer, return_tensors="tf")

        if shuffle:
            features.shuffle(100)

        features = features.batch(batch_size)
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
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr, epsilon=1e-08, clipnorm=1.0)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy('acc')
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path, monitor='val_acc',
                                                        save_weights_only=True, mode='max')

        features = self.get_features(x=x, y=y, batch_size=batch_size)
        self.model.fit(features,
                       epochs=epochs,
                       validation_split=val_split,
                       steps_per_epoch=int(len(features) / batch_size),
                       callbacks=[checkpoint])

    def predict(self, text):

        feature = self.tokenizer.encode_plus(text=text,
                                             return_tensors="tf")

        output = self.model(feature)[0]

        return output