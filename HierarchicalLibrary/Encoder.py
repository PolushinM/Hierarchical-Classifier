from typing import Union
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
from navec import Navec
from gensim import corpora
from gensim.models import LdaModel
from gensim.matutils import corpus2dense
import fasttext


class Encoder(object):

    def __init__(self) -> None:
        self.fasttext = None
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(emb)
        self.stop_words = set(nltk.corpus.stopwords.words('russian') + \
                              [',', '.', '', '|', ':', '"', '/', ')', '(', 'a', 'х', '(:', '):', ':(', ':)', 'и'])
        self.num_topics = None
        self.texts = []
        self.id_list = []
        self.dictionary = None
        self.lda = None
        self.PCA_matrix = None
        self.navec = None
        self.curr_dir = Path(".").parent
        return

    def tokenize_to_list(self, text: str) -> list:
        result = []
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            if token.text not in self.stop_words:
                result.append(token.text.lower())
        return result

    def lemmatize_to_list(self, text: str) -> list:
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

    def save_lda_model(self, name: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_lda_model.pickle']))
        self.lda.save(path)
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_dictionary.pickle']))
        with open(path, 'wb') as f:
            pickle.dump(self.dictionary, f)
        return

    def load_lda_model(self, name: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_lda_model.pickle']))
        self.lda = LdaModel.load(path, mmap='r')
        self.num_topics = self.lda.get_topics().shape[0]
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_dictionary.pickle']))
        with open(path, 'rb') as f:
            self.dictionary = pickle.load(f)
        return

    def load_navec_model(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        self.navec = Navec.load(path)
        return

    def load_fasttext_model(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        self.fasttext = fasttext.load_model(path)
        return

    def load_pca_matrix(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        with open(path, 'rb') as f:
            self.PCA_matrix = pickle.load(f)

    def save_pca_matrix(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        with open(path, 'wb') as f:
            pickle.dump(self.PCA_matrix, f)
        return

    def calc_PCA_matrix(self, dim: int = 128, words_max_buf_size: int = 300000, sample_size: int = 20000) -> np.array:
        count = 0
        for text in self.texts:
            count += len(text)
        count = min(count, words_max_buf_size)

        word_embs = np.ndarray((count, 300), dtype='float32')

        i = 0
        for text in self.texts:
            for word in text:
                embedding = self.navec.get(word)
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
        U, s, VT = np.linalg.svd(sample)
        self.PCA_matrix = VT[:dim, :]
        return VT[:dim, :]

    def fit_lda_model(self, num_topics: int, passes: int = 30, iterations: int = 10) -> None:

        self.num_topics = num_topics
        self.dictionary = corpora.Dictionary(self.texts)
        corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        np.random.seed(1)

        # Make a index to word dictionary.
        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        id2word = self.dictionary.id2token

        self.lda = LdaModel(corpus=corpus,
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

    def make_embeddings_dict(self, encoders: list) -> dict:

        embeddings = np.ndarray((len(self.texts), 0))
        for encoder in encoders:
            embeddings = np.hstack((embeddings, encoder(self.texts)))

        embeddings_dict = dict()
        i = 0
        for doc_id in self.id_list:
            embeddings_dict[doc_id] = embeddings[i]
            i += 1
        return embeddings_dict

    def get_embeddings(self, documents: list, encoders: list) -> np.array:
        texts = []
        for document in documents:
            texts.append(self.lemmatize_to_list(document))
        embeddings = np.ndarray((len(documents), 0))
        for encoder in encoders:
            embeddings = np.hstack((embeddings, encoder(texts)))
        return embeddings

    def lda_encoder(self, texts: list) -> np.array:
        embeddings = []
        for text in texts:
            embeddings.append(
                self.lda.get_document_topics(
                    self.dictionary.doc2bow(text)
                )
            )
        embeddings = corpus2dense(embeddings, self.num_topics).T
        embeddings = 2 * np.log(embeddings / (1 - embeddings) + 0.1) + 4.61
        return embeddings

    def doc2vec_encoder(self, texts: list, alpha: float) -> np.array:
        PCA_T = self.PCA_matrix.T
        dim = self.PCA_matrix.shape[0]
        result = np.ndarray((len(texts), dim))
        for i in range(len(texts)):
            doc_emb = np.zeros(dim, dtype='float32')
            for j in range(len(texts[i]) - 1, -1, -1):
                document = texts[i]
                embedding = self.navec.get(document[j])
                if embedding is not None:
                    doc_emb = embedding.dot(PCA_T) * alpha + doc_emb * (1 - alpha)
            result[i] = doc_emb / (np.linalg.norm(doc_emb) + 0.001) * dim ** 0.5
        return result

    def fasttext_encoder(self, texts: list) -> np.array:
        dim = self.fasttext.get_word_vector('').shape[0]
        result = np.ndarray((len(texts), dim))
        for i in range(len(texts)):
            doc_emb = self.fasttext.get_sentence_vector(' '.join(texts[i]))
            result[i] = doc_emb / (np.linalg.norm(doc_emb) + 0.001) * dim ** 0.5
        return result
