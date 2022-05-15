from __future__ import annotations
from typing import Union, List
import os
from pathlib import Path
import pickle
import tqdm
import pandas as pd
import numpy as np
from razdel import tokenize
import nltk
from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from gensim.utils import simple_preprocess


class TextProcessor(object):

    def __init__(self, add_stop_words: List[str] = None) -> None:
        self.dictionary = None
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(NewsEmbedding())
        if add_stop_words:
            self.stop_words = set(nltk.corpus.stopwords.words('russian') + add_stop_words)
        else:
            self.stop_words = set(nltk.corpus.stopwords.words('russian'))
        '''[',', '.', '', '|', ':', '"', '/', ')', '(', 'a', 'х', '(:', '):', ':(', ':)', 'и']'''
        self.texts: List[List[str]] = []
        self.id_list: List[int] = []
        self.curr_dir = Path(".").parent
        return

    def tokenize_to_list(self, text: str) -> List[str]:
        result = []
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            if token.text not in self.stop_words:
                result.append(token.text.lower())
        return result

    def lemmatize_to_list(self, text: str) -> List[str]:
        result = []
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        for token in doc.tokens:
            if token.lemma not in self.stop_words:
                result.append(token.lemma)
        return result

    def lemmatize_to_str(self, text: str) -> str:
        result = ''
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        for token in doc.tokens:
            if token.lemma not in self.stop_words:
                result = ' '.join([result, token.lemma])
        return result

    def lemmatize_data(self, data: pd.DataFrame, document_col: str, id_col: str) -> None:
        self.texts = []
        self.id_list = data[id_col].tolist()
        for text in tqdm.tqdm(data[document_col].tolist(), desc="Lemmatize", total=data.shape[0], mininterval=1):
            self.texts.append(self.lemmatize_to_list(text))
        return

    @classmethod
    def simple_lemmatize_to_list(cls, text: str) -> List[str]:
        return simple_preprocess(text)

    @classmethod
    def simple_lemmatize_to_str(cls, text: str) -> str:
        return ' '.join(simple_preprocess(text))

    def simple_lemmatize_data(self, data: pd.DataFrame, document_col: str, id_col: str) -> None:
        self.texts = []
        self.id_list = data[id_col].tolist()
        for text in tqdm.tqdm(data[document_col].tolist(), desc="Lemmatize", total=data.shape[0], mininterval=1):
            self.texts.append(self.simple_lemmatize_to_list(text))
        return

    def save_lemms_data(self, name: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_lemms.pickle']))
        with open(path, 'wb') as f:
            pickle.dump((self.texts, self.dictionary, self.id_list), f)
        return

    def load_lemms_data(self, name: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_lemms.pickle']))
        with open(path, 'rb') as f:
            self.texts, self.dictionary, self.id_list = pickle.load(f)
        return

    def make_embeddings_dict(self, encoders: List[object], verbose: bool = False) -> dict:

        embeddings = np.ndarray((len(self.texts), 0))
        i = 0
        for encoder in encoders:
            if verbose:
                print('Encoding:', f'{i}'.split('=')[0])
                i += 1
            embeddings = np.hstack((embeddings, encoder.transform(self.texts)))

        embeddings_dict = dict()
        i = 0
        for doc_id in self.id_list:
            embeddings_dict[doc_id] = embeddings[i]
            i += 1
        return embeddings_dict

    def get_embeddings(self, documents: list, encoders: List[object], simple_lemms: bool = False) -> np.array:
        texts = []
        for document in documents:
            if simple_lemms:
                texts.append(self.simple_lemmatize_to_list(document))
            else:
                texts.append(self.lemmatize_to_list(document))
        embeddings = np.ndarray((len(documents), 0))
        for encoder in encoders:
            embeddings = np.hstack((embeddings, encoder.transform(texts)))
        return embeddings