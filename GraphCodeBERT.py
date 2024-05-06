import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pandas as pd
from tqdm import tqdm
import networkx as nx
import numpy as np
import pydot
from transformers import AutoTokenizer, AutoModel


GCBtokenizer = AutoTokenizer.from_pretrained("FTGraphCodeBert.pt")
GCBmodel = AutoModel.from_pretrained("FTGraphCodeBert.pt")

def make_graph(f_path):
    graph = nx.DiGraph()
    with open(f_path, 'r') as f:
        dot_graph = f.read()

    dot_parsed = pydot.graph_from_dot_data(dot_graph)
    if dot_parsed:
        dot_graph = dot_parsed[0]
        for node in dot_graph.get_node_list():
            node_id = node.get_name().strip('"')
            label = node.get_label()
            if label:
                label_text = label.split('(')[1].split(')')[0]
                label_text = label_text.replace('\n', '').replace('  ', '')
                graph.add_node(node_id, label=label_text)

        for edge in dot_graph.get_edge_list():
            source = edge.get_source().strip('"')
            destination = edge.get_destination().strip('"')
            label_text = edge.get_label()
            if label_text:
                graph.add_edge(source, destination, label=label_text)

    mapping = {old_name: idx for idx, old_name in enumerate(graph.nodes())}
    G = nx.relabel_nodes(graph, mapping)

    return G


class GraphEmbedding():
    def __init__(self):
        self.model = GCBmodel
        self.tokenizer = GCBtokenizer
        self.embedding_graphs = {}

    def get_edge_index(self, graph):
        all_edge_index = []
        source_nodes = []
        target_nodes = []

        for source, target in graph.edges():
            source_nodes.append(source)
            target_nodes.append(target)

        all_edge_index.append(source_nodes)
        all_edge_index.append(target_nodes)

        return np.array(all_edge_index)

    def get_embeddings(self, graph: nx.DiGraph):
        for node, data in graph.nodes(data=True):
            word = data['label']
            # Tokenize your code
            code_tokens = self.tokenizer.tokenize(word)
            # Add special tokens
            tokens = [self.tokenizer.cls_token] + code_tokens + [self.tokenizer.sep_token]
            # Convert tokens to IDs
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Get embeddings
            feature = self.model(torch.tensor(tokens_ids)[None,:])[0][0,0]
            self.embedding_graphs[word] = feature

        results = np.array([value.detach().numpy() for value in self.embedding_graphs.values()])

        self.embedding_graphs = [(key, value.detach().numpy().tolist()) for key, value in self.embedding_graphs.items()]
        self.embedding_graphs = np.array(self.embedding_graphs, dtype=object)
        self.embedding_graphs = np.array(self.embedding_graphs)

        return results


class GCN(torch.nn.Module):
    def __init__(self, embeddings, hidden_channels, num_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.embeddings = embeddings
        self.mean_tensor = []
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        self.embeddings = x
        return x


    def get_node_embeddings(self):
        for i in range(self.embeddings.shape[0]):
            embedding_tensor = torch.from_numpy(self.embeddings[i])
            self.mean_tensor.append(torch.Tensor.mean(embedding_tensor))

        if self.embeddings.shape[0] >= 200:
            self.mean_tensor = self.mean_tensor[:200]
        else:
            for i in range(200-self.embeddings.shape[0]):
                self.mean_tensor.append(0)

        self.mean_tensor = np.array(self.mean_tensor)

        return self.mean_tensor
    

def gen_data(path):
    bad_sources = os.listdir(f'{path}/bad')
    bad_sources = [f'{path}/bad/{b}' for b in bad_sources]

    good_sources = os.listdir(f'{path}/good')
    good_sources = [f'{path}/good/{g}' for g in good_sources]

    sources = bad_sources + good_sources

    data = []

    for source in tqdm(sources):
        graph = make_graph(source)

        if len(graph.nodes()) == 0:
            continue

        embedding_model = GraphEmbedding()
        embeddings = embedding_model.get_embeddings(graph)
        edge_index = embedding_model.get_edge_index(graph)

        model = GCN(embeddings=embeddings, hidden_channels=128, num_node_features=200, num_classes=2)
        opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 5e-4)
        cr = torch.nn.CrossEntropyLoss()
        embeddings = torch.from_numpy(embeddings)
        edge_index = torch.from_numpy(edge_index)

        num_epochs=10
        for _ in range(num_epochs):
            model.train()
            opt.zero_grad()

        model.get_node_embeddings()
        data.append(model.mean_tensor)

    data = torch.from_numpy(np.array(data))
    df = pd.DataFrame(data.numpy())

    return df