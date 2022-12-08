import os
import re
import json
import numpy as np
from flask import Flask, request, jsonify
from langdetect import detect
import faiss
import nltk
import math
import torch
import torch.nn.functional as F
import string
from typing import Dict, List, Tuple, Union, Callable

# nltk.download('punkt')

FTOP = 50

STOPWORDS = [
    'myself',
    'our',
    'ours',
    'ourselves',
    'you',
    'your',
    'yours',
    'yourself',
    'yourselves',
    'him',
    'his',
    'himself',
    'she',
    'her',
    'hers',
    'herself',
    'its',
    'itself',
    'they',
    'them',
    'their',
    'theirs',
    'themselves',
    'what',
    'which',
    'who',
    'whom',
    'this',
    'that',
    'these',
    'those',
    'are',
    'was',
    'were',
    'been',
    'being',
    'have',
    'has',
    'had',
    'having',
    'do',
    'does',
    'did',
    'doing',
    'the',
    'and',
    'but',
    'because',
    'until',
    'while',
    'for',
    'with',
    'about',
    'against',
    'between',
    'into',
    'through',
    'during',
    'before',
    'after',
    'above',
    'below',
    'from',
    'down',
    'out',
    'off',
    'over',
    'under',
    'again',
    'further',
    'then',
    'once',
    'here',
    'there',
    'when',
    'where',
    'why',
    'how',
    'all',
    'any',
    'both',
    'each',
    'few',
    'more',
    'most',
    'other',
    'some',
    'such',
    'nor',
    'not',
    'only',
    'own',
    'same',
    'than',
    'too',
    'very',
    'can',
    'will',
    'just',
    'don',
    'should',
    'now'
]


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )


class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool = True, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        # shape = [B, L, D]
        embed_query = self.embeddings(query.long())
        # shape = [B, R, D]
        embed_doc = self.embeddings(doc.long())

        # shape = [B, L, R]
        matching_matrix = torch.einsum(
            'bld,brd->blr',
            F.normalize(embed_query, p=2, dim=-1),
            F.normalize(embed_doc, p=2, dim=-1)
        )
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out


class Ranker:

    def __init__(self):
        self.finit = False
        self.emb_path_glove = os.environ.get('EMB_PATH_GLOVE')
        self.emb_path_knrm = os.environ.get('EMB_PATH_KNRM')
        self.vocab_path =  os.environ.get('VOCAB_PATH')
        self.mlp_path = os.environ.get('MLP_PATH')
        self.findex = None
        self.status = None
        self.index_size = None
        self.max_len = 30
        self.model = KNRM(torch.load(self.emb_path_knrm)['weight'].numpy(), out_layers = [])
        self.model.mlp.load_state_dict(torch.load(self.mlp_path))

        with open(self.vocab_path, "r") as outfile:
            self.vocab = json.load(outfile)
        
        self.oov_val = self.vocab['OOV']

    def update_index(self, index):
        if not self.finit:
            self.index_size = len(index['documents'])
            self.findex = FaissIndex(index, self.emb_path_glove)
            self.finit = True
            self.status = 'ok'
        return {'status': self.status, 'index_size': self.index_size}
    
    def get_query(self, data):
        if not self.finit:
            return {'status': 'FAISS is not initialized!'}
        queries = data['queries']
        lang_check = []
        for query in queries:
            if detect(query) == 'en':
                lang_check.append(True)
            else:
                lang_check.append(False)
        suggestions = self.findex.get_top(data, k=FTOP)
        for i, query in enumerate(lang_check):
            if not query:
                suggestions[i] = None
        suggestions_ranked = []
        for qr, top in zip(queries, suggestions):
            if top is not None:
                top = self._top_ranking(qr, top, self.model)
            suggestions_ranked.append(top)
        return {'lang_check': lang_check, 'suggestions': suggestions_ranked}

    def _hadle_punctuation(self, inp_str):
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str

    def _simple_preproc(self, inp_str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self._hadle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)

    def _tokenized_text_to_index(self, tokenized_text):
        res = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
        return res[:self.max_len]

    def _top_ranking(self, query, top, model, top_qnt=10):
        qr = self._tokenized_text_to_index(self._simple_preproc(query))
        ans = []
        for i, text in enumerate(top):
            cnd = self._tokenized_text_to_index(self._simple_preproc(text[1]))
            data = {
                'query': torch.Tensor([qr]),
                'document': torch.Tensor([cnd])
            }
            with torch.no_grad():
                cfc = float(model.predict(data)[0][0])
            ans.append((top[i], cfc))
        ans.sort(key=lambda x: x[1], reverse=True)
        ans = [i[0] for i in ans][:top_qnt]
        return ans


class FaissIndex():

    def __init__(self, data, path):
        self.documents_index = data['documents']
        self.glove_vocab = self._read_glove_embeddings(path)
        self.embeddings_index = self._get_embeddings_index(
                self.documents_index,
                self.glove_vocab
            )
        self.faiss_index = self._get_faiss_index()

    def get_top(self, query_dict, k=10):
        queries = query_dict['queries']
        top_list = []
        for query in queries:
            tokens = self._cleaner(query)
            query_vector = self._get_embedding(self.glove_vocab, tokens)
            top_idxs = self.faiss_index.search(query_vector.reshape(1, -1).astype('float32'), k)[1][0]
            top = [(str(id), self.documents_index[str(id)]) for id in top_idxs]
            top_list.append(top)
        return top_list

    def _get_faiss_index(self):
        faiss_index = faiss.IndexIDMap(
                faiss.IndexFlatL2(list(self.embeddings_index.values())[0].shape[0])
            )
        faiss_index.add_with_ids(
            np.array(list(self.embeddings_index.values())).astype('float32'),
            np.array(list(self.embeddings_index.keys()))
            )
        return faiss_index

    def _get_embeddings_index(self, documents, vocab):
        return {
            int(idx): self._get_embedding(vocab, self._cleaner(text)) for idx, text \
            in zip(list(documents.keys()), list(documents.values()))
            }
    
    def _cleaner(self, text):
        clean = re.sub('[^a-z ]', ' ', str(text).lower())
        clean = re.sub(r" +", " ", clean).strip()
        tokens = [token for token in clean.split(' ') if (len(token) > 2 and token not in STOPWORDS)]
        return tokens

    def _get_embedding(self, vocab, tokens):
        embedding_list = [vocab[token] for token in tokens if token in vocab.keys()]
        if embedding_list:
            return np.mean(np.array(embedding_list), axis=0)
        else:
            return np.random.normal(1e-13, size=(list(vocab.values())[0].shape[0]))

    def _read_glove_embeddings(self, path):
        with open(path) as f:
            lines = [line.rstrip() for line in f]
        vocab = {}
        for line in lines:
            token, *vector = line.split(' ')
            vocab[token] = np.array(vector, dtype=np.float32)
        return vocab


app = Flask(__name__)

ranker = Ranker()

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok'})

@app.route('/query', methods=['POST'])
def query():
    data = json.loads(request.get_json())
    ans = ranker.get_query(data)
    return jsonify(ans)

@app.route('/update_index', methods=['POST'])
def update_index():
    index = json.loads(request.get_json())
    ans = ranker.update_index(index)
    return jsonify(ans)
