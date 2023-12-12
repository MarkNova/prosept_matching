import json
import re
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

def check_vectorizer_files(dirs=['./tokenizer/', './vectorizer/']):

    if not os.path.exists(dirs[0]):
        os.makedirs(dirs[0])
        tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        tokenizer.save_pretrained(dirs[0])
        
        
    if not os.path.exists(dirs[1]):
        os.makedirs(dirs[1])
        vectorizer = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        vectorizer.save_pretrained(dirs[1])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)


def remove_dots_except_between_numbers(string):
    new_string = ''
    for i, char in enumerate(string):
        if char in [',', '.']:
            if (string[i-1].isdigit() and
                i + 1 < len(string) and
                string[i+1].isdigit()):
                new_string += char
            else:
                new_string += ' '
                pass

        else:
            new_string += char
        previous_char = char

    return new_string


def replace_values(string):
    res = []
    
    string = remove_dots_except_between_numbers(string)
    
    splitted = string.split()
    
    for i, t in enumerate(splitted):
        if 'л' in t:
            value = t.replace('л', '')
            if value.replace(',', '.').replace('.', '').isdigit():
                value = float(value.replace(',', '.')) * 1000
                t = f'{int(value)} мл'
                res += [t]
                continue

            elif splitted[i - 1].replace(',', '.').replace('.', '').isdigit() and not 'мл' in t:
                value = float(splitted[i - 1].replace(',', '.')) * 1000
                t = f'{int(value)} мл'
                res = res[:-1]
                res += [t]
                continue
            else:
                pass
        if 'кг' in t:
            value = t.replace('кг', '')
            if value.replace(',', '.').replace('.', '').isdigit():
                value = float(value.replace(',', '.')) * 1000
                t = f'{int(value)} г'
                res += [t]
                continue

            elif splitted[i - 1].replace(',', '.').replace('.', '').isdigit():
                value = float(splitted[i - 1].replace(',', '.')) * 1000
                t = f'{int(value)} г'
                res = res[:-1]
                res += [t]
                continue
            else:
                pass
        
        res += [t]
    return ' '.join(res)


def string_filter_emb(string):
    
    string = string.lower() 
    string = re.sub(r'\d+-\d+', '', string)
    string = re.sub(r'\d+::\d+', '', string)
    string = re.sub(r'\d+:\d+', '', string)
    string = replace_values(string)
    string = re.sub(r'[^a-zo0-9а-я\s:]', ' ', string)
    string = re.sub(r'(?<=[а-я])(?=[a-z])|(?<=[a-z])(?=[а-я])', ' ', string)
    string = re.sub(r'(?<=[а-яa-z])(?=\d)|(?<=\d)(?=[а-яa-z])', ' ', string)
    
    string = string.replace(' 0 ', ' ')
    string = ' '.join([w for w in string.split()])
    return string


class InfloatVectorizer():
    def __init__(self,
                 toc_path='./tokenizer/',
                 vec_path='./vectorizer/'):
        
        check_vectorizer_files(dirs=[toc_path, vec_path])

        self.tokenizer = AutoTokenizer.from_pretrained(toc_path)
        self.model = AutoModel.from_pretrained(vec_path)

    def fit(self, X=None):
        pass

    def transform(self, corpus):
        batch_dict = self.tokenizer(
            corpus,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        last_hidden = outputs.last_hidden_state.masked_fill(
            ~batch_dict['attention_mask'][..., None].bool(), 0.0
        )
        embeddings = (last_hidden.sum(dim=1)
                      / batch_dict['attention_mask'].sum(dim=1)[..., None])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()


class DistanceRecommender():
    def __init__(self,
                 vectorizer,
                 simularity_func,
                 text_prep_func=string_filter_emb):
        self.vectorizer = vectorizer
        self.simularity_counter = simularity_func
        self.preprocessing = text_prep_func

    def fit(self,
            product_corpus,
            name_column,
            id_column,
            save_to_dir=False):
        preprocessed_corpus = (
            product_corpus[name_column].apply(
                self.preprocessing
            ).values.tolist()
        )
        self.vectorizer.fit(preprocessed_corpus)
        self.product_matrix = self.vectorizer.transform(preprocessed_corpus)
        self.product_index_to_id = {str(i): product_corpus.loc[i, id_column] for i in range(len(product_corpus))}
        if save_to_dir:
            
            if not os.path.exists('./model_files'):
                os.makedirs('./model_files')
            
            np.save('./model_files/product_matrix.npy', self.product_matrix)

            with open('./model_files/product_index_to_id.json', 'w') as file:
                json.dump(self.product_index_to_id, file, cls=NumpyEncoder)

    def from_pretrained(
        self,
        product_matrix_path='./model_files/product_matrix.npy',
        product_index_to_id_dict_path='./model_files/product_index_to_id.json'
    ):
        self.product_matrix = np.load(product_matrix_path)

        with open(product_index_to_id_dict_path, 'rb') as file:
            self.product_index_to_id = json.load(file)

    def recommend(self,
                  dealer_corpus: list[dict]):
        dealer_corpus = pd.Series(dealer_corpus)

        dealer_corpus = dealer_corpus.apply(
            self.preprocessing
        ).values.tolist()
        vectors = self.vectorizer.transform(dealer_corpus)
        sims = self.simularity_counter(vectors, self.product_matrix)

        result = []
        for vec in sims:
            result += [[self.product_index_to_id[str(index)] for index in vec.argsort()[::-1]]]
        return np.array(result)


def dealerprice_table(table_path='marketing_dealerprice.csv',
                      product_id_column='product_key',
                      dealer_id_column='dealer_id',
                      read_params={'on_bad_lines': "skip",
                                   'encoding': 'utf-8',
                                   'sep': ';'}):
    '''
    Функция принимает:
    .Путь к csv файлу, содержащему результаты парсинга.
    .Названия колонок с id товаров и id дилеров
    .Параметры чтения csv можно указать, если вдруг они изменятся.
    '''

    table_csv = pd.read_csv(table_path, **read_params)
    table_csv = table_csv.sort_values(
        'date', ascending=False
    ).drop_duplicates(
        subset=[
            product_id_column,
            dealer_id_column
        ]
    )
    return table_csv

def prossept_products_table(table_path='marketing_product.csv',
                            product_names_column = 'name',
                            read_params={'on_bad_lines': "skip",
                                           'encoding': 'utf-8',
                                           'sep': ';'}
                             ):
    '''
    Функция принимает путь к csv файлу, содержащему актуальную информацию по товарам заказчика.
    Дополнительно указывается название колонки с внутренними неймингами для удаления плохих строк.
    '''
    
    table_csv = pd.read_csv(table_path, **read_params) 
    table_csv = table_csv.dropna(subset='name')
    table_csv = table_csv.reset_index(drop=True)
    
    return table_csv

def names_join_ozon(x):
    total = []
    if type(x['name']) == str:
        total += [x['name'].strip()]
    if type(x['ozon_name']) == str:
        total += [x['ozon_name'].strip()]
    return ' '.join(total)