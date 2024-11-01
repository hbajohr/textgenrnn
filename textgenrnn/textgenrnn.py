import json
import re

import numpy as np
import tensorflow as tf
import tqdm
from pkg_resources import resource_filename
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

# Remove the v1 compatibility imports
class textgenrnn:
    META_TOKEN = '<s>'
    config = {
        'rnn_layers': 2,
        'rnn_size': 128,
        'rnn_bidirectional': False,
        'max_length': 40,
        'max_words': 10000,
        'dim_embeddings': 100,
        'word_level': False,
        'single_text': False
    }
    default_config = config.copy()

    def __init__(self, weights_path=None,
                 vocab_path=None,
                 config_path=None,
                 name="textgenrnn",
                 allow_growth=None):

        if weights_path is None:
            weights_path = resource_filename(__name__,
                                             'textgenrnn_weights.hdf5')

        if vocab_path is None:
            vocab_path = resource_filename(__name__,
                                           'textgenrnn_vocab.json')

        if allow_growth is not None:
            # Updated GPU memory growth setting for TF 2.x
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        if config_path is not None:
            with open(config_path, 'r',
                      encoding='utf8', errors='ignore') as json_file:
                self.config = json.load(json_file)

        self.config.update({'name': name})
        self.default_config.update({'name': name})

        with open(vocab_path, 'r',
                  encoding='utf8', errors='ignore') as json_file:
            self.vocab = json.load(json_file)

        self.tokenizer = Tokenizer(filters='', lower=False, char_level=True)
        self.tokenizer.word_index = self.vocab
        self.num_classes = len(self.vocab) + 1
        self.model = textgenrnn_model(self.num_classes,
                                      cfg=self.config,
                                      weights_path=weights_path)
        self.indices_char = dict((self.vocab[c], c) for c in self.vocab)

    # Rest of the methods remain the same...
