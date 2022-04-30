from typing import Union, Tuple
import os
from pathlib import Path
import pickle
import tqdm
import pandas as pd
import numpy as np
from collections import Iterable


class CategoryTree(object):

    def __init__(self) -> None:
        self.tree = dict()
        self.proba_cache = dict()
        self.curr_dir = Path(".").parent
        return

    def __getitem__(self, node_id: int) -> None:
        return self.tree[node_id]

    def __len__(self) -> int:
        return len(self.tree)

    def keys(self) -> Iterable:
        return self.tree.keys()

    def items(self) -> Iterable:
        return self.tree.items()

    def save_tree(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        with open(path, 'wb') as f:
            pickle.dump(self.tree, f)
        return

    def load_tree(self, fname: str, directory: str = '') -> None:
        path = os.path.join(self.curr_dir, directory, fname)
        with open(path, 'rb') as f:
            self.tree = pickle.load(f)

    def add_nodes_from_df(self, data: pd.DataFrame, parent_id_col: str, title_col: str) -> None:
        for key, row in data.iterrows():
            self.tree[key] = {'parent_id': row[parent_id_col],
                              'children_ids': set(),
                              'title': row[title_col],
                              'ancestral_titles': set(),
                              'embedding': np.array([]),
                              'weights': np.array([]),
                              'intercept': 0.,
                              'good_count': 0,
                              'good_ids': []
                              }
        self.update_ancestors()
        return

    def add_goods_from_df(self, data: pd.DataFrame, category_id_col: str, good_id_col: str) -> None:
        for key, row in data.iterrows():
            self.add_good(row[category_id_col], row[good_id_col])
        return

    def add_good(self, cat_id: int, good_id: int) -> None:
        curr_id = cat_id
        while True:
            self.tree[curr_id]['good_ids'].append(good_id)
            self.tree[curr_id]['good_count'] += 1
            next_id = self[curr_id]['parent_id']
            if next_id == 0:
                break
            self.tree[curr_id]['ancestral_titles'].add(self[next_id]['title'])
            self.tree[curr_id]['ancestral_titles'].update(self[next_id]['ancestral_titles'])
            self.tree[next_id]['children_ids'].add(curr_id)
            curr_id = next_id
        return

    def update_ancestors(self) -> None:
        for i in range(3):
            for curr_id in self.keys():
                next_id = self.tree[curr_id]['parent_id']
                if next_id == 0 or next_id not in self.keys():
                    continue
                self.tree[curr_id]['ancestral_titles'].add(self[next_id]['title'])
                self.tree[curr_id]['ancestral_titles'].update(self[next_id]['ancestral_titles'])
        return

    def update_embeddings(self, embeddings_dict: dict) -> None:
        dim = embeddings_dict[next(iter(embeddings_dict))].shape[0]
        for _, node in self.items():
            if node['good_count'] > 0:
                node['embedding'] = np.zeros((dim,), dtype='float32')
                for good_id in node['good_ids']:
                    node['embedding'] += embeddings_dict[good_id] / node['good_count']
        for _, node in self.items():
            node['embedding'] = node['embedding'] / (np.linalg.norm(node['embedding']) + 0.001)
        return

    def mix_in_description_embs(self, encoder: callable, weight: int = 4) -> None:
        id_list = list(self.keys())
        path_titles = []
        for node_id in id_list:
            path_title = [self[node_id]['title']] + list(self[node_id]['ancestral_titles']).copy()
            if 'Все категории' in path_title:
                path_title.remove('Все категории')
            path_titles.append(' '.join(path_title))
        title_embeddings = encoder(path_titles)
        for node_id, embedding in zip(id_list, title_embeddings):
            if self.tree[node_id]['embedding'].shape[0] == 0:
                self.tree[node_id]['embedding'] = embedding
            else:
                self.tree[node_id]['embedding'] = (self.tree[node_id]['embedding'] * (
                    self.tree[node_id]['good_count']) ** 0.5
                                                   + embedding * weight) / (
                                                          (self.tree[node_id]['good_count']) ** 0.5 + weight)
        return

    def get_siblings_ids(self, node_id: int) -> Union[set, None]:
        if (node_id not in self.tree) or (self[node_id]['good_count'] == 0):
            return None
        parent = self[node_id]['parent_id']
        if parent == 0:
            return None
        childs = self[parent]['children_ids'].copy()
        if not childs:
            return None
        childs.remove(node_id)
        return childs

    def choose_leaf(self, classifier: callable, good_embedding: np.array = None, document: str = None) -> int:
        current_id = 1
        next_id = 1
        while next_id:
            current_id = next_id
            next_id, _ = self.choose_child(current_id,
                                           classifier=classifier,
                                           good_embedding=good_embedding,
                                           document=document)
        self.proba_cache = dict()
        return current_id

    def choose_leaf_proba(self, classifier: callable,
                          good_embedding: np.array = None,
                          document: str = None) -> Tuple[int, float]:
        next_id = 1
        current_id = 1
        leaf_proba = 1.
        while next_id:
            current_id = next_id
            next_id, proba = self.choose_child(current_id,
                                               classifier=classifier,
                                               good_embedding=good_embedding,
                                               document=document)
            leaf_proba *= proba
        self.proba_cache = dict()
        return current_id, leaf_proba

    def choose_leaf_proba_upwards(self, leaf_ids_probas: list,
                                  classifier: callable,
                                  embeddings_dict: dict, proba_power: float = 1) -> Tuple[int, float]:
        max_proba = 0.
        best_leaf = 1
        for leaf_id, leaf_proba in leaf_ids_probas:
            next_node = leaf_id['parent_id']
            proba = leaf_proba
            while next_node > 1:
                proba *= self.eval_node_proba(node_id=next_node,
                                              classifier=classifier,
                                              good_embedding=embeddings_dict[next_node]) ** proba_power
                next_node = next_node['parent_id']
            if proba > max_proba:
                max_proba = proba
                best_leaf = leaf_id
        return best_leaf, max_proba

    def choose_child(self, node_id: int, classifier: callable,
                     good_embedding: np.array = None,
                     document: str = None) -> Union[None, tuple]:

        childs = self[node_id]['children_ids']
        if len(childs) == 0:
            return None, 1.
        if len(childs) == 1:
            return list(childs)[0], 1.0
        max_proba = 0.
        best_child = None
        for child in childs:
            proba = self.eval_node_full_proba(child,
                                              classifier=classifier,
                                              good_embedding=good_embedding,
                                              document=document)
            if max_proba < proba:
                max_proba = proba
                best_child = child
        return best_child, max_proba

    def eval_node_full_proba(self, node_id: int,
                             good_embedding: np.array = None,
                             document: str = None,
                             classifier: callable = None) -> float:

        self_proba = self.eval_node_proba(node_id,
                                          classifier=classifier,
                                          good_embedding=good_embedding,
                                          document=document)

        childs = self[node_id]['children_ids']
        if len(childs) == 0:
            return self_proba
        max_proba = 0.
        sum_proba = 0.
        for child in childs:
            proba = self.eval_node_proba(child,
                                         classifier=classifier,
                                         good_embedding=good_embedding,
                                         document=document)
            sum_proba += proba ** 2
            if max_proba < proba:
                max_proba = proba
        return (max_proba * 0.7 + sum_proba ** 0.5 * 0.3) * self_proba

    def eval_node_proba(self, node_id: int,
                        classifier: callable,
                        good_embedding: np.array = None,
                        document: str = None) -> float:

        if node_id in self.proba_cache:
            return self.proba_cache[node_id]
        elif document:
            proba = classifier.predict_local_proba_by_doc(node_id, document)
            print(proba)
        elif len(self[node_id]['weights']) == 0:
            return 1e-8
        else:
            proba = classifier.predict_local_proba(self[node_id]['weights'],
                                                   self[node_id]['intercept'],
                                                   self[node_id]['embedding'],
                                                   good_embedding)
        self.proba_cache[node_id] = proba
        return proba

    def get_id_path(self, node_id: int) -> list:
        path = []
        current_id = node_id
        while current_id > 1:
            path.append(current_id)
            current_id = self[current_id]['parent_id']
        return path

    def get_id_path_set(self, node_id: int) -> set:
        path = set()
        current_id = node_id
        while current_id > 1:
            path.add(current_id)
            current_id = self[current_id]['parent_id']
        return path

    def hF1_score(self, true_leafs: list, pred_leafs: list) -> Union[None, float]:
        if len(true_leafs) != len(pred_leafs):
            return None
        sum_Ti = 0
        sum_Pi = 0
        sum_Ti_Pi = 0
        for true, pred in zip(true_leafs, pred_leafs):
            pr_rec_tuple = self.get_good_pr_rec(true, pred)
            sum_Ti += pr_rec_tuple[0]
            sum_Pi += pr_rec_tuple[1]
            sum_Ti_Pi += pr_rec_tuple[2]

        hP = sum_Ti_Pi / sum_Pi
        hR = sum_Ti_Pi / sum_Ti
        return 2 * hP * hR / (hP + hR)

    def hF1_score_01(self, true_leafs, pred_leafs) -> Union[None, float]:
        if len(true_leafs) != len(pred_leafs):
            return None
        sum_Ti = 0
        sum_Pi = 0
        sum_Ti_Pi = 0
        for true, pred in zip(true_leafs, pred_leafs):
            pr_rec_tuple = self.get_good_pr_rec(true, pred)
            sum_Ti += pr_rec_tuple[0] + 2
            sum_Pi += pr_rec_tuple[1] + 2
            sum_Ti_Pi += pr_rec_tuple[2] + 2

        hP = sum_Ti_Pi / sum_Pi
        hR = sum_Ti_Pi / sum_Ti
        return 2 * hP * hR / (hP + hR)

    def get_good_pr_rec(self, good_category_id: int, predicted_leaf_id: int) -> tuple:
        Ti_set = self.get_id_path_set(good_category_id)
        Pi_set = self.get_id_path_set(predicted_leaf_id)
        inter_set = Ti_set.intersection(Pi_set)
        return len(Ti_set), len(Pi_set), len(inter_set)

    def fit_local_weights(self, classifier: object, embeddings_dict: dict, C: int = 1,
                          reg_count_power: float = 0.4, verbose=False) -> None:
        for key in tqdm.tqdm(list(self.keys()), total=len(self)):
            siblings = self.get_siblings_ids(key)
            if siblings:
                siblings_goods = []
                for sibling in siblings:
                    siblings_goods += self[sibling]['good_ids']

                dim = embeddings_dict[next(iter(embeddings_dict))].shape[0] + 1
                if self.tree[key]['weights'].shape[0] == dim:
                    warm_coefs = self.tree[key]['weights']
                else:
                    warm_coefs = np.zeros(dim)
                self.tree[key]['weights'], \
                self.tree[key]['intercept'] = classifier.fit_node_weights(self[key]['good_ids'],
                                                                          siblings_goods,
                                                                          self[key]['embedding'],
                                                                          embeddings_dict,
                                                                          C_in=C,
                                                                          reg_count_power=reg_count_power,
                                                                          warm_coefs=warm_coefs,
                                                                          warm_intercept=self.tree[key]['intercept'],
                                                                          verbose=verbose)
        return
