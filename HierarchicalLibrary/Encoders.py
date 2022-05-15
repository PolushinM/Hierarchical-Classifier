from __future__ import annotations
from typing import Union, List
from abc import ABC, abstractmethod
import os
from pathlib import Path
import pickle
import tqdm
import pandas as pd
import numpy as np

from navec import Navec
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import corpus2dense

import fasttext
import torch
from transformers import AutoTokenizer, AutoModel


class BaseEncoder(ABC):
    def __init__(self) -> None:
        self.curr_dir: object = Path(".").parent
        self.model: object = None
        return

    @abstractmethod
    def transform(self, texts: List[List[str]]) -> np.array:
        return

    @abstractmethod
    def load_model(self, name: str, directory: str = '') -> None:
        return


class LdaEncoder(BaseEncoder):

    def __init__(self) -> None:
        super().__init__()
        self.num_topics = None
        self.dictionary = None
        return

    def save_model(self, name: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, ''.join([name, '_lda_model.pickle']))
        self.model.save(path)
        path = os.path.join(str(self.curr_dir), directory, ''.join([name, '_dictionary.pickle']))
        with open(path, 'wb') as f:
            pickle.dump(self.dictionary, f)
        return

    def load_model(self, name: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, ''.join([name, '_lda_model.pickle']))
        self.model = LdaModel.load(path, mmap='r')
        self.num_topics = self.model.get_topics().shape[0]
        path = os.path.join(str(self.curr_dir), directory, ''.join([name, '_dictionary.pickle']))
        with open(path, 'rb') as f:
            self.dictionary = pickle.load(f)
        return

    def fit(self, texts: List[List[str]], num_topics: int, passes: int = 30, iterations: int = 10) -> None:
        self.num_topics = num_topics
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        np.random.seed(1)

        # Make an index to word dictionary.
        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        id2word = self.dictionary.id2token

        self.model = LdaModel(corpus=corpus,
                              id2word=id2word,
                              chunksize=400000,
                              alpha='auto',
                              eta='auto',
                              iterations=iterations,
                              num_topics=num_topics,
                              passes=passes,
                              eval_every=None,
                              random_state=1
                              )
        return

    def transform(self, texts: List[List[str]]) -> np.array:
        embeddings = []
        for text in texts:
            embeddings.append(
                self.model.get_document_topics(
                    self.dictionary.doc2bow(text)
                )
            )
        embeddings = corpus2dense(embeddings, self.num_topics).T
        embeddings = 2 * np.log(embeddings / (1 - embeddings) + 0.1) + 4.61
        return embeddings


class NavecEncoder(BaseEncoder):

    def __init__(self, alpha: float, dim: [None | int] = None) -> None:
        super().__init__()
        self.alpha = alpha
        self.dim = dim
        self.PCA: np.array = None
        return

    def load_model(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        self.model = Navec.load(path)
        return

    def load_pca(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        with open(path, 'rb') as f:
            self.PCA = pickle.load(f)

    def save_pca(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        with open(path, 'wb') as f:
            pickle.dump(self.PCA, f)
        return

    def calc_pca(self, texts: List[List[str]], words_max_buf_size: int = 300000, sample_size: int = 20000) -> np.array:
        count = 0
        for text in texts:
            count += len(text)
        count = min(count, words_max_buf_size)

        word_embs = np.ndarray((count, 300), dtype='float32')

        i = 0
        for text in texts:
            for word in text:
                embedding = self.model.get(word)
                if embedding is not None:
                    word_embs[i] = embedding
                    i += 1
                    if i == count:
                        break
            if i == count:
                break

        sample = word_embs[np.random.choice(word_embs.shape[0],
                                            min(sample_size, words_max_buf_size),
                                            replace=False)].astype('float32')
        U, s, self.PCA = np.linalg.svd(sample)
        return self.PCA

    def transform(self, texts: List[List[str]]) -> np.array:
        # Set embedding dimension
        if self.dim and self.dim <= 300:
            PCA_T = self.PCA[:self.dim, :].T
        else:
            PCA_T = np.eye(300)

        result = np.ndarray((len(texts), self.dim))
        for i in range(len(texts)):
            doc_emb = np.zeros(self.dim, dtype='float32')
            for j in range(len(texts[i]) - 1, -1, -1):
                document = texts[i]
                embedding = self.model.get(document[j])
                if embedding is not None:
                    doc_emb = embedding.dot(PCA_T) * self.alpha + doc_emb * (1 - self.alpha)
            result[i] = doc_emb / (np.linalg.norm(doc_emb) + 0.001) * self.dim ** 0.5
        return result


class FasttextEncoder(BaseEncoder):

    def __init__(self, alpha: [None | float] = None) -> None:
        super().__init__()
        self.alpha = alpha
        return

    def load_model(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        self.model = fasttext.load_model(path)
        return

    def transform(self, texts: List[List[str]]) -> np.array:
        dim = self.model.get_word_vector('').shape[0]
        result = np.ndarray((len(texts), dim))
        if self.alpha:
            for i in range(len(texts)):
                document = texts[i]
                doc_emb = self.model.get_sentence_vector(' '.join(document))
                for j in range(len(texts[i]) - 1, -1, -1):
                    embedding = self.model.get_word_vector(document[j])
                    doc_emb = embedding * self.alpha + doc_emb * (1 - self.alpha)
                result[i] = doc_emb / (np.linalg.norm(doc_emb) + 0.001) * dim ** 0.5
        else:
            for i in range(len(texts)):
                doc_emb = self.model.get_sentence_vector(' '.join(texts[i]))
                result[i] = doc_emb * dim ** 0.5
        return result


class BertEncoder(BaseEncoder):

    def __init__(self, dim: [None | int] = None) -> None:
        super().__init__()
        self.PCA: np.array = None
        self.dim = dim
        self.tokenizer = None
        self.model = None
        return

    def load_model(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        return

    def load_pca(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        with open(path, 'rb') as f:
            self.PCA = pickle.load(f)

    def save_pca(self, fname: str, directory: str = '') -> None:
        path = os.path.join(str(self.curr_dir), directory, fname)
        with open(path, 'wb') as f:
            pickle.dump(self.PCA, f)
        return

    def calc_pca(self, texts: List[List[str]], sample_size: int = 20000) -> np.array:
        count = min(len(texts), sample_size)
        sample = np.ndarray((count, self.get_bert_embedding('').shape[0]), dtype='float32')

        i = 0
        for text in texts:
            sample[i] = self.get_bert_embedding(' '.join(text))
            i += 1
            if i == count:
                break

        U, s, self.PCA = np.linalg.svd(sample)
        return self.PCA

    def transform(self, texts: List[List[str]]) -> np.array:
        # Set embedding dimension
        if self.dim and self.dim <= self.get_bert_embedding('').shape[0]:
            PCA_T = self.PCA[:self.dim, :].T
        else:
            PCA_T = np.eye(self.get_bert_embedding('').shape[0])
            self.dim = self.get_bert_embedding('').shape[0]
        result = np.ndarray((len(texts), self.dim))
        for i in range(len(texts)):
            doc_emb = self.get_bert_embedding(' '.join(texts[i])).dot(PCA_T)
            result[i] = doc_emb * self.dim ** 0.5
        return result

    def get_bert_embedding(self, text: str) -> np.array:
        t = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**{k: v.to(self.model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()
    