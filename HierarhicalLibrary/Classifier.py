from __future__ import annotations
from typing import Union, Tuple, Any
import math
import os
from pathlib import Path
from random import choice
import pickle
import tqdm
import pandas as pd
import numpy as np
import catboost as cb
from sklearn.linear_model import LogisticRegressionCV
from time import time


class Classifier(object):

    def __init__(self, catboost_parameters: [dict | None] = None, tol: float = 0.05, max_iter: int = 100) -> None:
        if catboost_parameters is None:
            catboost_parameters = {'loss_function': 'Logloss',
                                   'iterations': 100,
                                   'depth': 7,
                                   'rsm': 1.0,
                                   'random_seed': 1,
                                   'custom_metric': 'F1',
                                   'learning_rate': 0.5
                                   }

        self.global_classifier = cb.CatBoostClassifier(n_estimators=catboost_parameters['iterations'],
                                                       learning_rate=catboost_parameters['learning_rate'],
                                                       rsm=catboost_parameters['rsm'],
                                                       depth=catboost_parameters['depth'],
                                                       loss_function=catboost_parameters['loss_function'],
                                                       thread_count=-1,
                                                       random_seed=0,
                                                       logging_level='Silent')
        self.max_iter = max_iter
        self.tol = tol
        self.use_global_classifier = False
        self.concat_array = None
        self.target_array = None
        self.curr_dir = Path(".").parent
        self.fasttext = None
        self.fasttext_num_labels = 20
        self.fasttext_proba_cache = dict()
        self.fasttext_cache_document = ''
        return

    def save_global_classifier(self, fname: str, format: str = 'cbm', directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        self.global_classifier.save_model(path, format=format)

    def load_global_classifier(self, fname: str, format: str = 'cbm', directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        self.global_classifier.load_model(path, format=format)
        self.use_global_classifier = True
        return

    def load_fasttext_model(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        self.fasttext = fasttext.load_model(path)
        return

    def predict_fasttext_labels(self, text: str) -> zip[Tuple[int, float]]:
        """Predict number of most likely leaf labels and probabilities for it by fasttext model"""
        predict = self.fasttext.predict(text, k=self.fasttext_num_labels)
        labels_tuple = predict[0]
        proba_array = predict[1]
        labels_array = np.ndarray((self.fasttext_num_labels,), dtype='int16')
        for i in range(self.fasttext_num_labels):
            labels_array[i] = int(labels_tuple[i][9:])
        return zip(labels_array, proba_array)

    def predict_fasttext_leaf_proba(self, node_id: int, document: str) -> float:
        """Returns leaf node proba from cached fasttext prediction"""
        if document != self.fasttext_cache_document:
            self.fasttext_cache_document = document
            self.fasttext_proba_cache = dict(self.predict_fasttext_labels(document))
        if node_id in self.fasttext_proba_cache:
            return self.fasttext_proba_cache[node_id]
        else:
            return 1e-8

    def fit_node_weights(self, goods: list,
                         siblings_goods: list,
                         node_embedding: np.array,
                         embeddings_dict: dict,
                         C_in: float, reg_count_power: float,
                         warm_coefs: np.array, warm_intercept: float,
                         verbose: bool = False
                         ) -> tuple[np.array, float]:
        additional_features = 1
        dim = embeddings_dict[next(iter(embeddings_dict))].shape[0]
        positives = 0
        negatives = 0
        train_array = []
        target = np.ndarray((0, 1))
        begin_time = time()
        # Add positive examples
        for good in goods:
            cos_sim = node_embedding.dot(embeddings_dict[good]) / dim
            if self.use_global_classifier:
                global_logit = self.predict_logit_global(node_embedding, embeddings_dict[good])
                train_array.append(np.hstack((embeddings_dict[good], np.array([cos_sim, global_logit]))))
                additional_features = 2
            else:
                train_array.append(np.hstack((embeddings_dict[good], np.array([cos_sim]))))
            target = np.append(target, 1)
            positives += 1
        begin_neg_time = time()
        # Add negative siblings examples
        for good in siblings_goods:
            cos_sim = node_embedding.dot(embeddings_dict[good]) / dim
            if self.use_global_classifier:
                global_logit = self.predict_logit_global(node_embedding, embeddings_dict[good])
                train_array.append(np.hstack((embeddings_dict[good], np.array([cos_sim, global_logit]))))
                additional_features = 2
            else:
                train_array.append(np.hstack((embeddings_dict[good], np.array([cos_sim]))))
            target = np.append(target, 0)
            negatives += 1
        begin_neg_random_time = time()
        # Add negative random examples
        keys = list(embeddings_dict.keys())
        goods_set = set(goods)
        for i in range(negatives):
            good = goods[0]
            while good in goods_set:
                good = choice(keys)
            cos_sim = node_embedding.dot(embeddings_dict[good]) / dim
            if self.use_global_classifier:
                global_logit = self.predict_logit_global(node_embedding, embeddings_dict[good])
                train_array.append(np.hstack((embeddings_dict[good], np.array([cos_sim, global_logit]))))
                additional_features = 2
            else:
                train_array.append(np.hstack((embeddings_dict[good], np.array([cos_sim]))))
            target = np.append(target, 0)
            negatives += 1

        # Return standard values in case of data deficiency
        if positives == 0:
            return np.zeros(dim + additional_features), -5.
        elif negatives == 0:
            return np.zeros(dim + additional_features), 5.
        elif positives < 2:
            return np.zeros(dim + additional_features), -2.
        elif negatives < 2:
            return np.zeros(dim + additional_features), 2

        # Calculate parameters
        count = min(positives, negatives)
        class_weights = {0: (positives / (positives + negatives)) ** 0.2,
                         1: (negatives / (positives + negatives)) ** 0.2 * 2}
        begin_CV_time = time()
        C = C_in * count ** reg_count_power
        # Fit logistic regression
        local_model = LogisticRegressionCV(Cs=[C / 64, C / 4, C * 4, C * 64],
                                           cv=2,
                                           tol=self.tol,
                                           random_state=1,
                                           scoring='neg_log_loss',
                                           solver='saga',
                                           max_iter=self.max_iter,
                                           multi_class='ovr',
                                           class_weight=class_weights,
                                           n_jobs=-1)
        local_model.fit(np.random.rand(4, dim + additional_features), [1, 0, 1, 0])
        local_model.coef_ = np.array([warm_coefs])
        local_model.intercept_ = np.array([warm_intercept])
        local_model.fit(np.array(train_array), target)

        # Print some technical info
        if verbose:
            score = (local_model.scores_[1][0] * 0.5 + local_model.scores_[1][1] * 0.5).max()
            print(
                f'Count={count}, C_mult={local_model.C_[0] / C:.3f}, C={local_model.C_[0]:.3f}, score={score:.3f}')
            print(f'pos: {begin_neg_time - begin_time:.1f}, neg_sib: {begin_neg_random_time - begin_neg_time:.1f}, '
                  f'neg_rand: {begin_CV_time - begin_neg_random_time:.1f}, CV: {time() - begin_CV_time:.1f}')

        return local_model.coef_[0], local_model.intercept_[0]

    def predict_local_proba(self, weights: np.array,
                            intercept: float,
                            node_embedding: np.array,
                            good_embedding: np.array
                            ) -> float:
        """Returns probability estimation of node membership for certain commodity with considering global classifier logit,
        cosine similarity and local classifier logit"""
        dim = good_embedding.shape[0]
        cos_sim = node_embedding.dot(good_embedding) / dim
        if self.use_global_classifier:
            global_logit = self.predict_logit_global(node_embedding, good_embedding)
            embedding = np.hstack((good_embedding, np.array([cos_sim, global_logit])))
        else:
            embedding = np.hstack((good_embedding, np.array([cos_sim])))
        logit = weights.dot(embedding) + intercept
        return 1 / (1 + math.exp(-logit))

    def predict_leaf_proba_by_doc(self, leaf_node_id: int, document: str) -> float:
        return self.predict_fasttext_node_proba(leaf_node_id, document)

    def fit_global_classifier(self) -> None:
        self.global_classifier.fit(cb.Pool(self.concat_array, label=self.target_array))
        self.use_global_classifier = True
        return

    def predict_logit_global(self, node_embedding, good_embedding) -> float:
        return self.global_classifier.predict(np.hstack((node_embedding, good_embedding)),
                                              prediction_type='RawFormulaVal')

    def calc_global_train_array(self, embeddings_dict: object, train_tree: dict) -> None:
        """Calculation of dataset for training of global classifier"""
        dim = embeddings_dict[next(iter(embeddings_dict))].shape[0]

        train_set_list = []

        for key in list(train_tree.keys()):
            buf_dict = dict()
            siblings = train_tree.get_siblings_ids(key)

            if siblings:  # TODO: fix, use arrays instead of dict
                # Add positive examples
                node = train_tree[key]
                for good in node['good_ids']:
                    buf_dict['node_emb'] = node['embedding']
                    buf_dict['good_emb'] = embeddings_dict[good]
                    buf_dict['target'] = 1
                    train_set_list.append(buf_dict.copy())
                # Add negative examples
                for sib_id in siblings:
                    node = train_tree[sib_id]
                    for good in node['good_ids']:
                        buf_dict['node_emb'] = train_tree[key]['embedding']
                        buf_dict['good_emb'] = embeddings_dict[good]
                        buf_dict['target'] = 0
                        train_set_list.append(buf_dict.copy())

        self.concat_array = np.ndarray((len(train_set_list), dim * 2), dtype='float16')
        self.target_array = np.ndarray((len(train_set_list), 1), dtype='int16')
        i = 0
        for item in train_set_list:
            self.concat_array[i] = np.hstack((item['node_emb'], item['good_emb']))
            self.target_array[i] = item['target']
            i += 1
        return

    def save_global_train_array(self, name, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_global_array.pickle']))
        with open(path, 'wb') as f:
            pickle.dump((self.concat_array, self.target_array), f)
        return

    def load_global_train_array(self, name, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, ''.join([name, '_global_array.pickle']))
        with open(path, 'rb') as f:
            self.concat_array, self.target_array = pickle.load(f)
        return

    def delete_global_train_array(self) -> None:
        self.concat_array = None
        self.target_array = None
        return
