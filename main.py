import copy
import os
from functools import reduce
from multiprocessing import Pool
from bisect import insort

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')


def tokenize(text):
    wnl = WordNetLemmatizer()
    sw = set(stopwords.words('english'))

    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    tokens = filter(lambda w: w.isalpha() and len(w) > 1, tokens)
    tokens = filter(lambda w: w not in sw, tokens)
    tokens = [wnl.lemmatize(word) for word in tokens]
    return tokens


def merge_index(index1, index2):
    index = copy.deepcopy(index1)
    for key, val_list in index2.items():
        if key not in index:
            index[key] = val_list
        else:
            for val in val_list:
                insort(index[key], val)
    return index


def worker(file_path, doc_id):
    with open(file_path, 'r') as f:
        data = f.read().replace('\n', '')
    return {token: [doc_id] for token in tokenize(data)}


def build_inverted_index(doc_dir):
    doc_id_map = {}
    id_counter = 1

    files = []
    for file_name in os.listdir(doc_dir):
        file_path = os.path.join(doc_dir, file_name)
        files.append((file_path, id_counter))
        doc_id_map[id_counter] = file_name[:-4]
        id_counter += 1

    with Pool(5) as p:
        list_dicts = p.starmap(worker, files)

    inv_index = reduce(merge_index, list_dicts)

    return inv_index, doc_id_map


if __name__ == "__main__":
    document_dir = "./Documents"
    inverted_index, document_ids = build_inverted_index(document_dir)
    for term, postings in inverted_index.items():
        print(f"{term}: {postings}")
    print(document_ids)
