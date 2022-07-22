
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as gnn

class GraphAgent(nn.Module):

    def __init__(self, nemb, nvec, out_per_stem, out_per_mol, num_conv_steps, mdp_cfg, version='v1'):
        '''
        nemb=256
        nvec=0
        out_per_stem=mdp.num_blocks
        out_per_mol=1
        num_conv_steps=10
        version='v4'
        '''
        super().__init__()
        print(version)
        if version == 'v5': version = 'v4'
        self.version = version
        self.embeddings = nn.ModuleList([
            # Embedding(size of dictionary of embeddings, size of each embedding vector)
            nn.Embedding(mdp_cfg.num_true_blocks + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types + 1, nemb),
            nn.Embedding(mdp_cfg.num_stem_types, nemb)])
        self.conv = gnn.NNConv(nemb, nemb, nn.Sequential(), aggr='mean') # this is a message passing NN
        nvec_1 = nvec * (version == 'v1' or version == 'v3')
        nvec_2 = nvec * (version == 'v2' or version == 'v3')
        self.block2emb = nn.Sequential(nn.Linear(nemb + nvec_1, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb)) # Linear(input_sample_size, output_sample_size): Applies a linear transformation to the incoming data: y = xA^T + by
        self.gru = nn.GRU(nemb, nemb)
        self.stem2pred = nn.Sequential(nn.Linear(nemb * 2 + nvec_2, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                       nn.Linear(nemb, out_per_stem))
        self.global2pred = nn.Sequential(nn.Linear(nemb, nemb), nn.LeakyReLU(),
                                         nn.Linear(nemb, out_per_mol))
        #self.set2set = Set2Set(nemb, processing_steps=3)
        self.num_conv_steps = num_conv_steps
        self.nemb = nemb
        self.training_steps = 0
        self.categorical_style = 'softmax'
        self.escort_p = 6

        # self defined layers for new MPNN implementation
        self.nodelayer1 = nn.Linear(256, 256)
        self.nodelayer2 = nn.Linear(256, 256) # pass the sum of node embedding + the output from all neighbours


    def forward2(self, graph_data, vec_data=None, do_stems=True):
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x) # embed x 
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes) # embed stemtypes
        graph_data.edge_attr = bondemb(graph_data.edge_attr) # embed edge_attr
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2)) 
        out = graph_data.x
        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        elif self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out) # embed blocks in x using the Sequential block2emb

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr)) # passes the graph into the MPNN and then leaky_relu
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous()) # contiguous(): Returns a contiguous in memory tensor containing the same data as self tensor
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            stem_block_batch_idx = (
                torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
                + graph_data.stems[:, 0])
            if self.version == 'v1' or self.version == 'v4':
                stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
            elif self.version == 'v2' or self.version == 'v3':
                stem_out_cat = torch.cat([out[stem_block_batch_idx],
                                          graph_data.stemtypes,
                                          vec_data[graph_data.stems_batch]], 1)

            stem_preds = self.stem2pred(stem_out_cat) # get the stem predictions from the stem data
        else:
            stem_preds = None
        mol_preds = self.global2pred(gnn.global_mean_pool(out, graph_data.batch)) # get the mol predictions from the graph data
        return stem_preds, mol_preds

    # alternative forward method with self implemented MPNN
    def forward(self, graph_data, vec_data=None, do_stems=True):
        # forward method
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x) # embed x 

        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes) # embed stemtypes

        graph_data.edge_attr = bondemb(graph_data.edge_attr) # embed edge_attr
        
        # sum the from and to embeddings for each edge
        # use sum instead of concat as it needs to be commutative
        # e.g. [0,5] and [5,0] will have different edge_embedding if concat is used
        graph_data.edge_attr = (graph_data.edge_attr[:, 0] + graph_data.edge_attr[:, 1])
        
        out = graph_data.x

        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        elif self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out) # embed blocks in x using the Sequential block2emb
        
        h = out.unsqueeze(0)

        # TODO: implement MPNN from ground up
        l = []
        for i in range(graph_data.x.size(0)): # iterate through each node in the molecule
            # iterate through its neighbours 
            neighbours = torch.zeros(256, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            neighboursIndices = self.getAllNeighbours(i, graph_data.edges) # get the edge indices of all current node's neighbours
            for idx, isFlipped in neighboursIndices:
                temp = (graph_data.edge_attr[idx])
                # sum the edge attr embedding with the to node embedding
                toIndex = graph_data.edges[idx][0 if isFlipped else 1] # get the index of the neighbouring node
                temp = temp + F.leaky_relu(self.nodelayer1(out[toIndex])) # pass the node through a linear layer
                neighbours = neighbours + temp # sum up all the current nodes neighbours

            neighbours = neighbours / (len(neighboursIndices) if len(neighboursIndices) else 1) # get average of all neighbours
            # sum with the current node embedding
            newSum = neighbours + out[i]
            l.append(F.leaky_relu(self.nodelayer2(newSum)))
        out = torch.stack(l)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        if do_stems:
            stem_block_batch_idx = (
                torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
                + graph_data.stems[:, 0])
            stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)

            stem_preds = self.stem2pred(stem_out_cat) # get the stem predictions from the stem data
        else:
            stem_preds = None

        # get the mol predictions from the graph data
        mol_preds = self.global2pred(gnn.global_mean_pool(out, graph_data.batch))
        return stem_preds, mol_preds

    # takes in the index of the node and the edgelist
    # returns the index of the edges that are connected to the specified node
    def getAllNeighbours(self, index, edges):
        res = []
        for i in range(len(edges)):
            edge = edges[i]
            fromNode = edge[0]
            toNode = edge[1]
            if fromNode == index:
                res.append((i, False))  # boolean to indicate if the edge is flipped
            elif toNode == index:
                res.append((i, True))
        return res

    # get the stem_out_cat variable
    def get_stem_out_cat(self, graph_data, vec_data=None, do_stems=True):
        blockemb, stememb, bondemb = self.embeddings
        graph_data.x = blockemb(graph_data.x) # embed x 
        if do_stems:
            graph_data.stemtypes = stememb(graph_data.stemtypes) # embed stemtypes
        graph_data.edge_attr = bondemb(graph_data.edge_attr) # embed edge_attr
        graph_data.edge_attr = (
            graph_data.edge_attr[:, 0][:, :, None] * graph_data.edge_attr[:, 1][:, None, :]
        ).reshape((graph_data.edge_index.shape[1], self.nemb**2)) 
        out = graph_data.x
        if self.version == 'v1' or self.version == 'v3':
            batch_vec = vec_data[graph_data.batch]
            out = self.block2emb(torch.cat([out, batch_vec], 1))
        elif self.version == 'v2' or self.version == 'v4':
            out = self.block2emb(out) # embed blocks in x using the Sequential block2emb

        h = out.unsqueeze(0)

        for i in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv(out, graph_data.edge_index, graph_data.edge_attr)) # passes the graph into the MPNN and then leaky_relu
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous()) # contiguous(): Returns a contiguous in memory tensor containing the same data as self tensor
            out = out.squeeze(0)

        # Index of the origin block of each stem in the batch (each
        # stem is a pair [block idx, stem atom type], we need to
        # adjust for the batch packing)
        stem_block_batch_idx = (
            torch.tensor(graph_data.__slices__['x'], device=out.device)[graph_data.stems_batch]
            + graph_data.stems[:, 0])
        stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_data.stemtypes], 1)
        print('stem_block_batch_idx: ', stem_block_batch_idx)
        return stem_out_cat

    def out_to_policy(self, s, stem_o, mol_o):
        if self.categorical_style == 'softmax':
            stem_e = torch.exp(stem_o)
            mol_e = torch.exp(mol_o[:, 0])
        elif self.categorical_style == 'escort':
            stem_e = abs(stem_o)**self.escort_p
            mol_e = abs(mol_o[:, 0])**self.escort_p
        Z = gnn.global_add_pool(stem_e, s.stems_batch).sum(1) + mol_e + 1e-8
        return mol_e / Z, stem_e / Z[s.stems_batch, None]

    # implement negative log-likelihood to use as the loss function
    def action_negloglikelihood(self, s, a, g, stem_o, mol_o):
        mol_p, stem_p = self.out_to_policy(s, stem_o, mol_o)
        #print(Z.shape, Z.min().item(), Z.mean().item(), Z.max().item())
        mol_lsm = torch.log(mol_p + 1e-20)
        stem_lsm = torch.log(stem_p + 1e-20)
        #print(mol_lsm.shape, mol_lsm.min().item(), mol_lsm.mean().item(), mol_lsm.max().item())
        #print(stem_lsm.shape, stem_lsm.min().item(), stem_lsm.mean().item(), stem_lsm.max().item(), '--')
        return -self.index_output_by_action(s, stem_lsm, mol_lsm, a)

    def index_output_by_action(self, s, stem_o, mol_o, a):
        stem_slices = torch.tensor(s.__slices__['stems'][:-1], dtype=torch.long, device=stem_o.device)
        return (
            stem_o[stem_slices + a[:, 1]][
                torch.arange(a.shape[0]), a[:, 0]] * (a[:, 0] >= 0)
            + mol_o * (a[:, 0] == -1))

    def sum_output(self, s, stem_o, mol_o):
        return gnn.global_add_pool(stem_o, s.stems_batch).sum(1) + mol_o

def mol2graph(mol, mdp, floatX=torch.float, bonds=False, nblocks=False):
    f = lambda x: torch.tensor(x, dtype=torch.long, device=mdp.device)
    if len(mol.blockidxs) == 0:
        data = Data(# There's an extra block embedding for the empty molecule
            x=f([mdp.num_true_blocks]),
            edge_index=f([[],[]]),
            edge_attr=f([]).reshape((0,2)),
            stems=f([(0,0)]),
            stemtypes=f([mdp.num_stem_types]), # also extra stem type embedding
            edges = f([]))
        return data
    edges = [(i[0], i[1]) for i in mol.jbonds]
    #edge_attrs = [mdp.bond_type_offset[i[2]] +  i[3] for i in mol.jbonds]
    t = mdp.true_blockidx
    if 0:
        edge_attrs = [((mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[2]) * mdp.num_stem_types +
                       (mdp.stem_type_offset[t[mol.blockidxs[i[1]]]] + i[3]))
                      for i in mol.jbonds]
    else: # defining the attributes/features of each jbond/edge
        # stem_type_offset is a list of the cumulative summation of the largest r group for each block
        edge_attrs = [(mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[2], # feature 1: the offset of the from block
                       mdp.stem_type_offset[t[mol.blockidxs[i[1]]]] + i[3]) # feature 2: the offset of the to block
                      for i in mol.jbonds]
    # Here stem_type_offset is a list of offsets to know which
    # embedding to use for a particular stem. Each (blockidx, atom)
    # pair has its own embedding.
    stemtypes = [mdp.stem_type_offset[t[mol.blockidxs[i[0]]]] + i[1] for i in mol.stems]

    data = Data(x=f([t[i] for i in mol.blockidxs]), # index of blocks in the molecule
                edge_index=f(edges).T if len(edges) else f([[],[]]), # a tensor representing the edge list which contains the from list and the to list
                edge_attr=f(edge_attrs) if len(edges) else f([]).reshape((0,2)), # a tensor containing the features of the j bonds/edges
                stems=f(mol.stems) if len(mol.stems) else f([(0,0)]), # a tensor representing all the available stems 
                stemtypes=f(stemtypes) if len(mol.stems) else f([mdp.num_stem_types]), # a tensor containing the embedded representation of the available stems
                edges=f(edges) if len(edges) else f([]))
    data.to(mdp.device)
    assert not bonds and not nblocks
    #print(data)
    return data


def mols2batch(mols, mdp):
    batch = Batch.from_data_list(
        mols, follow_batch=['stems'])
    batch.to(mdp.device)
    return batch
