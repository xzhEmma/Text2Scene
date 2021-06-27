#!/usr/bin/env python

import math, cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from abstract_utils import conv3x3 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
#from miscc.config import cfg
 
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class TextEncoder(nn.Module):
    def __init__(self, db):
        super(TextEncoder, self).__init__()
        self.db = db
        self.cfg = db.cfg       
        self.nfeat = 600
        self.nhid = 600
        self.output_dim = 1024

        self.gcn = GCN(self.nfeat, self.nhid, self.output_dim)
        self.embedding = nn.Embedding(self.cfg.input_vocab_size, self.cfg.n_embed)#2538 300
        if self.cfg.emb_dropout_p > 0:
            self.embedding_dropout = nn.Dropout(p=self.cfg.emb_dropout_p)

        rnn_cell = self.cfg.rnn_cell.lower()
        if rnn_cell == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.rnn = self.rnn_cell(self.cfg.n_embed, self.cfg.n_src_hidden, 
            self.cfg.n_rnn_layers, batch_first=True, 
            bidirectional=self.cfg.bidirectional, 
            dropout=self.cfg.rnn_dropout_p)

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)
        #print(self.db.lang_vocab.vectors.size())#[10,300]
        self.embedding.weight.data.copy_(self.db.lang_vocab.vectors)
        

    def init_hidden(self, bsize):
        num_layers = self.cfg.n_rnn_layers
        hsize = self.cfg.n_src_hidden
        num_directions = 2 if self.cfg.bidirectional else 1

        hs = torch.zeros(num_layers * num_directions, bsize, hsize)
        if self.cfg.cuda:
            hs = hs.cuda()

        if self.cfg.rnn_cell.lower() == 'lstm':
            cs = torch.zeros(num_layers * num_directions, bsize, hsize)
            if self.cfg.cuda:
                cs = cs.cuda()
            return (hs, cs)
        
        return hs

    def forward(self, input_inds, input_lens,x, adj):
        """
        Args:
            - **input_inds**  (bsize, slen) or (bsize, 3, slen)
            - **input_msks**  (bsize, slen) or (bsize, 3, slen)
        Returns: dict containing
            - **output_feats**   (bsize, tlen, hsize)
            - **output_embed**   (bsize, tlen, esize)
            - **output_msks**    (bsize, tlen)
            - **output_hiddens** [list of](num_layers * num_directions, bsize, hsize)
        """
        bsize, n_seg, slen = input_inds.size() #3个句子，句子最大长度

        out_embs, out_rfts, out_msks = [], [], []
        out_hids, out_cels = [], []

        factor = 2 if self.cfg.bidirectional else 1
        hsize  = factor * self.cfg.n_src_hidden
        pad_rft = torch.zeros(1, 1, hsize)
        pad_emb = torch.zeros(1, 1, self.cfg.n_embed)
        if self.cfg.cuda:
            pad_rft = pad_rft.cuda()
            pad_emb = pad_emb.cuda()

        for i in range(bsize):
            inst_rfts, inst_embs, inst_hids = [], [], []
            for j in range(n_seg):
                # every segment has its own hidden states
                # curr_hidden = self.init_hidden(1)
                curr_len  = input_lens[i, j].view(-1).data.item() #4
                curr_inds = input_inds[i, j].view(-1) #tensor [  3,   5, 112, 242,   2,   0],
                curr_inds = curr_inds[:curr_len] #tensor [  3,   5, 112, 242]
                curr_inds = curr_inds.view(1, curr_len) #tensor[[  3,   5, 112, 242]]
                #print('inds',curr_inds)
                curr_vecs = self.embedding(curr_inds)
                #print(curr_vecs.size())
                #curr_obj_vecs = self.embedding(curr_obj_inds)
                if self.cfg.emb_dropout_p > 0:
                    curr_vecs = self.embedding_dropout(curr_vecs)
                inst_embs.append(curr_vecs)
                
                # curr_feats, curr_hidden = self.rnn(curr_vecs, curr_hidden)
                curr_feats, curr_hidden = self.rnn(curr_vecs)
                inst_rfts.append(curr_feats)
                inst_hids.append(curr_hidden)
            
            inst_rfts = torch.cat(inst_rfts, 1)
            inst_embs = torch.cat(inst_embs, 1)
            
            tlen = inst_rfts.size(1)
            n_pad = n_seg * slen - tlen
            # print('n_seg', n_seg)
            # print('slen', slen)
            # print('tlen', tlen)
            # Pad mask
            inst_msks = [1.0] * tlen
            if n_pad > 0:
                inst_msks = inst_msks + [0.0] * n_pad
            inst_msks = np.array(inst_msks)
            inst_msks = torch.from_numpy(inst_msks).float()
            if self.cfg.cuda:
                inst_msks = inst_msks.cuda()

            if n_pad > 0:
                # Pad features 
                inst_rfts = torch.cat([inst_rfts, pad_rft.expand(1, n_pad, hsize)], 1)
                inst_embs = torch.cat([inst_embs, pad_emb.expand(1, n_pad, self.cfg.n_embed)], 1)
            
           # print('inst_rfts: ', inst_rfts.size())
            # print('inst_embs: ', inst_embs.size())
            # print('inst_msks: ', inst_msks.size())

            out_msks.append(inst_msks)
            out_rfts.append(inst_rfts)
            out_embs.append(inst_embs)

            # Average hiddens
            if isinstance(inst_hids[0], tuple):
                hs = torch.stack([inst_hids[0][0], inst_hids[1][0], inst_hids[2][0]], 0)
                cs = torch.stack([inst_hids[0][1], inst_hids[1][1], inst_hids[2][1]], 0)
                out_hids.append(hs)
                out_cels.append(cs)
            else:
                hs = torch.stack([inst_hids[0], inst_hids[1], inst_hids[2]], 0)
                out_hids.append(hs)

        out_rfts = torch.cat(out_rfts, 0).contiguous()
        out_embs = torch.cat(out_embs, 0).contiguous()
        out_hids = torch.cat(out_hids, 2).contiguous()
        out_msks = torch.stack(out_msks, 0).contiguous()

        print('out_rfts: ', out_rfts.size())
        # print('out_embs: ', out_embs.size())
        # print('out_hids: ', out_hids.size())
        vecs = []
        # x [4, 18, 2538]
        if x.dim()==2:
            x=x.unsqueeze(0)
        else:
            x=x
        for w in x:  
           # print('w',w.size())
            ps=[]
            count = 0
            combine= []
            for i in w: # w [18, 2538]
                i_list = i.cpu().numpy().tolist() 
                # if 1 in i_list:
               # print('i',i.size())
                
                p=[i_list.index(x) for x in i_list if x==1] 
                            
                if len(p)!=2:
                    combine.append(0)    
                else:
                    count =count+1
                    combine.append(1) 
                for x in p:
                    ps.append(x ) 
                        
            ps_tensor = torch.Tensor(ps).long()
            ps_tensor = ps_tensor.cuda()
            vec = self.embedding(ps_tensor)
            vec = torch.nn.ZeroPad2d(padding=(0,300,0,0))(vec)
            k = [combine.index(x) for x in combine if x==1]
            if len(k)!=0:
                for s in k:        
                    vec[s][300:600:1] =  vec[s+1][0:300:1]
                    vec[s+1][0:300:1] = torch.IntTensor(1, 300).zero_()
##################################################
            else:
                vec = vec
            if vec.size(0)!=18:
                vec = torch.nn.ZeroPad2d(padding=(0,0,0,18-vec.size(0)))(vec)
            
            vecs.append(vec)
            x = torch.stack(vecs,0)

        out_sg = self.gcn(x,adj) #[4,10,1024]
        out = {}
        out['rfts'] = out_rfts
        out['embs'] = out_embs
        out['msks'] = out_msks
        out['sg'] = out_sg
        #print(out['sg'])
        if len(out_cels) > 0:
            out_cels = torch.cat(out_cels, 2).contiguous()
            out_last_hids = []
            for i in range(out_hids.size(0)):
                out_last_hids.append((out_hids[i], out_cels[i]))
            out['hids'] = out_last_hids
        else:
            out_last_hids = []
            for i in range(out_hids.size(0)):
                out_last_hids.append(out_hids[i])
            out['hids'] = out_last_hids
        return out


class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.cfg = config

        original_model = models.resnet50(pretrained=True)
        self.conv1   = original_model.conv1 
        self.bn1     = original_model.bn1
        self.relu    = original_model.relu 
        self.maxpool = original_model.maxpool
        self.layer1  = original_model.layer1
        self.layer2  = original_model.layer2
        self.layer3  = original_model.layer3
        # self.layer4  = original_model.layer4
        self.upsample = nn.Upsample(size=(self.cfg.grid_size[1], self.cfg.grid_size[0]), mode='bilinear', align_corners=True)

        # for param in self.parameters():
        #     param.requires_grad = False

    def forward(self, stack_imgs):
        if self.cfg.finetune_lr == 0:
            self.eval()
        # if self.cfg.teacher_forcing:
        #     bsize, slen, fsize, height, width = stack_imgs.size()
        #     inputs = stack_imgs.view(bsize * slen, fsize, height, width)
        # else:
        #     bsize, fsize, height, width = stack_imgs.size()
        #     slen = 1
        #     inputs = stack_imgs
        bsize, slen, fsize, height, width = stack_imgs.size()
        inputs = stack_imgs.view(bsize * slen, fsize, height, width)
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        x = self.upsample(x)
        nsize, fsize, gh, gw = x.size()
        assert(nsize == bsize * slen)
        x = x.view(bsize, slen, fsize, gh, gw)
        return x 


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output_dim):
        #600 600 1024
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output_dim)
        
        #self.linear = nn.Linear(tgt_dim, tgt_dim, bias=True)
    def forward(self, x, adj):    
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

#  batch, len, dim = output.size()
       
#         uh = nn.Linear(len, 10, bias=True)
#         uh = uh.view(batch, 1, len, dim)
#         output = nn.Tanh(uh)
#         print(output.size())
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True, CUDA=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        # print('in_features',in_features)
        # print('out_features',out_features)
        self.out_features = out_features
        
        if CUDA:
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            if CUDA:
                self.bias = Parameter(torch.cuda.FloatTensor(out_features))
            else:
                self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
  
        
    def forward(self, input, adj, CUDA=True):
        # support = torch.mm(input, self.weight)
        if CUDA:
            input = input.cuda()
            adj = adj.cuda()
        else:
            input=input
            adj=adj     
        support = torch.matmul(input, self.weight)
        support = support.float()
        adj = adj.float()
        # output = torch.spmm(adj, support)
        output = torch.matmul(adj,support)#[4*18*600]
       
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



