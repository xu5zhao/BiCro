"""SGRAF model"""
import math
from collections import OrderedDict

import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.clip_grad import clip_grad_norm_
from sklearn.mixture import GaussianMixture

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_sim(x1, x2, dim=-1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class EncoderImage(nn.Module):
    """
    Build local region representations by common-used FC-layer.
    Args: - images: raw local detected regions, shape: (batch_size, 36, 2048).
    Returns: - img_emb: finial local region embeddings, shape:  (batch_size, 36, 1024).
    """

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer"""
        r = np.sqrt(6.0) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        img_emb = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb

    def load_state_dict(self, state_dict):
        """Overwrite the default one to accept state_dict from Full model"""
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)


class EncoderText(nn.Module):
    """
    Build local word representations by common-used Bi-GRU or GRU.
    Args: - images: raw local word ids, shape: (batch_size, L).
    Returns: - img_emb: final local word embeddings, shape: (batch_size, L, 1024).
    """

    def __init__(
        self,
        vocab_size,
        word_dim,
        embed_size,
        num_layers,
        use_bi_gru=False,
        no_txtnorm=False,
    ):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.cap_rnn = nn.GRU(
            word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru
        )

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, lengths):
        """Handles variable size captions"""
        # embed word ids to vectors
        cap_emb = self.embed(captions)
        cap_emb = self.dropout(cap_emb)

        # pack the caption
        packed = pack_padded_sequence(
            cap_emb, lengths, batch_first=True, enforce_sorted=False
        )

        # forward propagate RNN
        out, _ = self.cap_rnn(packed)

        # reshape output to (batch_size, hidden_size)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)

        if self.use_bi_gru:
            cap_emb = (
                cap_emb[:, :, : cap_emb.size(2) // 2]
                + cap_emb[:, :, cap_emb.size(2) // 2 :]
            ) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


class VisualSA(nn.Module):
    """
    Build global image representations by self-attention.
    Args: - local: local region embeddings, shape: (batch_size, 36, 1024)
          - raw_global: raw image by averaging regions, shape: (batch_size, 1024)
    Returns: - new_global: final image by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate, num_region):
        super(VisualSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(num_region),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local regions and raw global image
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, 36)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final image, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class TextSA(nn.Module):
    """
    Build global text representations by self-attention.
    Args: - local: local word embeddings, shape: (batch_size, L, 1024)
          - raw_global: raw text by averaging words, shape: (batch_size, 1024)
    Returns: - new_global: final text by self-attention, shape: (batch_size, 1024).
    """

    def __init__(self, embed_dim, dropout_rate):
        super(TextSA, self).__init__()

        self.embedding_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Dropout(dropout_rate)
        )
        self.embedding_common = nn.Sequential(nn.Linear(embed_dim, 1))

        self.init_weights()
        self.softmax = nn.Softmax(dim=1)

    def init_weights(self):
        for embeddings in self.children():
            for m in embeddings:
                if isinstance(m, nn.Linear):
                    r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                    m.weight.data.uniform_(-r, r)
                    m.bias.data.fill_(0)
                elif isinstance(m, nn.BatchNorm1d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, local, raw_global):
        # compute embedding of local words and raw global text
        l_emb = self.embedding_local(local)
        g_emb = self.embedding_global(raw_global)

        # compute the normalized weights, shape: (batch_size, L)
        g_emb = g_emb.unsqueeze(1).repeat(1, l_emb.size(1), 1)
        common = l_emb.mul(g_emb)
        weights = self.embedding_common(common).squeeze(2)
        weights = self.softmax(weights)

        # compute final text, shape: (batch_size, 1024)
        new_global = (weights.unsqueeze(2) * local).sum(dim=1)
        new_global = l2norm(new_global, dim=-1)

        return new_global


class GraphReasoning(nn.Module):
    """
    Perform the similarity graph reasoning with a full-connected graph
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_sgr: reasoned graph nodes after several steps, shape: (batch_size, L+1, 256)
    """

    def __init__(self, sim_dim):
        super(GraphReasoning, self).__init__()

        self.graph_query_w = nn.Linear(sim_dim, sim_dim)
        self.graph_key_w = nn.Linear(sim_dim, sim_dim)
        self.sim_graph_w = nn.Linear(sim_dim, sim_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def forward(self, sim_emb):
        sim_query = self.graph_query_w(sim_emb)
        sim_key = self.graph_key_w(sim_emb)
        sim_edge = torch.softmax(torch.bmm(sim_query, sim_key.permute(0, 2, 1)), dim=-1)
        sim_sgr = torch.bmm(sim_edge, sim_emb)
        sim_sgr = self.relu(self.sim_graph_w(sim_sgr))
        return sim_sgr

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class AttentionFiltration(nn.Module):
    """
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """

    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(
            torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1
        )
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class EncoderSimilarity(nn.Module):
    """
    Compute the image-text similarity by SGR, SAF, AVE
    Args: - img_emb: local region embeddings, shape: (batch_size, 36, 1024)
          - cap_emb: local word embeddings, shape: (batch_size, L, 1024)
    Returns:
        - sim_all: final image-text similarities, shape: (batch_size, batch_size).
    """

    def __init__(self, embed_size, sim_dim, module_name="AVE", sgr_step=3):
        super(EncoderSimilarity, self).__init__()
        self.module_name = module_name

        self.v_global_w = VisualSA(embed_size, 0.4, 36)
        self.t_global_w = TextSA(embed_size, 0.4)

        self.sim_tranloc_w = nn.Linear(embed_size, sim_dim)
        self.sim_tranglo_w = nn.Linear(embed_size, sim_dim)

        self.sim_eval_w = nn.Linear(sim_dim, 1)
        self.sigmoid = nn.Sigmoid()

        if module_name == "SGR":
            self.SGR_module = nn.ModuleList(
                [GraphReasoning(sim_dim) for i in range(sgr_step)]
            )
        elif module_name == "SAF":
            self.SAF_module = AttentionFiltration(sim_dim)
        else:
            raise ValueError("Invalid module")

        self.init_weights()
    def glo_emb(self,img_emb, cap_emb, cap_lens):
        cap_emb_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        #img_glo = self.v_global_w(img_emb, img_ave) #(batch_size, 1024)
        img_glo = img_ave

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            #cap_glo_i = self.t_global_w(cap_i, cap_ave_i)  #(batch_size, 1024)
            cap_emb_all.append(cap_ave_i)

        # (n_image, n_caption)
        cap_emb_all = torch.cat(cap_emb_all, 0)

        return img_glo,cap_emb_all
    def forward(self, img_emb, cap_emb, cap_lens):
        sim_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        img_glo = self.v_global_w(img_emb, img_ave)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            cap_glo_i = self.t_global_w(cap_i, cap_ave_i)

            # local-global alignment construction
            Context_img = SCAN_attention(cap_i_expand, img_emb, smooth=9.0)
            sim_loc = torch.pow(torch.sub(Context_img, cap_i_expand), 2)
            sim_loc = l2norm(self.sim_tranloc_w(sim_loc), dim=-1)

            sim_glo = torch.pow(torch.sub(img_glo, cap_glo_i), 2)
            sim_glo = l2norm(self.sim_tranglo_w(sim_glo), dim=-1)

            # concat the global and local alignments
            sim_emb = torch.cat([sim_glo.unsqueeze(1), sim_loc], 1)

            # compute the final similarity vector
            if self.module_name == "SGR":
                for module in self.SGR_module:
                    sim_emb = module(sim_emb)
                sim_vec = sim_emb[:, 0, :]
            else:
                sim_vec = self.SAF_module(sim_emb)

            # compute the final similarity score
            sim_i = self.sigmoid(self.sim_eval_w(sim_vec))
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.0) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def SCAN_attention(query, context, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn * smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0,warmup_rate=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.warmup_rate = warmup_rate
    def forward(
        self,
        scores,
        hard_negative=True,
        labels=None,
        soft_margin="linear",
        mode="train",
        noise_tem = 0.9,
    ):
        # compute image-sentence score matrix
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        if labels is None:
            margin = self.margin
        elif soft_margin == "linear":
            margin = self.margin * labels
        elif soft_margin == "exponential":
            s = (torch.pow(10, labels) - 1) / 9
            margin = self.margin * s
        elif soft_margin == "sin":
            s = torch.sin(math.pi * labels - math.pi / 2) / 2 + 1 / 2
            margin = self.margin * s

        # compare every diagonal score to scores in its column: caption retrieval
        #cost_s = (margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row: image retrieval
        cost_im = (margin + scores - d2).clamp(min=0)
        if labels is not None and soft_margin == "exponential":
            margin = margin.t()
        # compare every diagonal score to scores in its column: caption retrieval
        cost_s = (margin + scores - d1).clamp(min=0)
        # clear diagonals
        mask = torch.eye(scores.size(0)) > 0.5
        mask = mask.to(cost_s.device)
        cost_s, cost_im = cost_s.masked_fill_(mask, 0), cost_im.masked_fill_(mask, 0)

        # maximum and mean
        cost_s_max, cost_im_max = cost_s.max(1)[0], cost_im.max(0)[0]
        cost_s_mean, cost_im_mean = cost_s.mean(1), cost_im.mean(0)

        if mode == "predict":
            p = margin - (cost_s_mean + cost_im_mean) / 2
            p = p.clamp(min=0, max=margin)
            idx = torch.argsort(p)
            ratio = scores.size(0) // 10 + 1
            p = p / torch.mean(p[idx[-ratio:]])
            return p

        if mode == "predict_clean":
            p = margin - (cost_s_mean + cost_im_mean) / 2
            p = p.clamp(min=0, max=margin)
            idx = torch.argsort(p)
            ratio = scores.size(0) // 10 + 1
            #p = p / torch.mean(p[idx[-ratio:]])
            return idx[-ratio:],idx[:-ratio]

        elif mode == "warmup_sele":
            all_loss = cost_s_mean + cost_im_mean
            y = all_loss.topk(k=int(scores.size(0)*self.warmup_rate), dim=0, largest=False, sorted=True)
            index = torch.zeros(scores.size(0)).cuda()
            index[y[1]]=1
            all_loss = all_loss*index
            #选择clean样本
            return all_loss.sum()
        elif mode == "noise_hard":
            #labels
            index = labels>noise_tem
            if hard_negative:
                return ((cost_s_max + cost_im_max)*index).sum()
            else:
                return ((cost_s_mean+ cost_im_mean)*index).sum()
        elif mode =='warmup':
            return cost_s_mean.sum() + cost_im_mean.sum()
        elif mode == "train" or mode == "noise_soft":
            if hard_negative:
                return cost_s_max.sum() + cost_im_max.sum()
            else:
                return cost_s_mean.sum() + cost_im_mean.sum()

        elif mode == "eval_loss" or mode == "y_score":
            return cost_s_mean + cost_im_mean

def SIM_PAIR(clean_input,noise_input, eps=1e-8):
    """
    
    """
    clean_input_norm = torch.norm(clean_input,p=2,dim=1).unsqueeze(0)
    noise_input_norm = torch.norm(noise_input,p=2,dim=1).unsqueeze(1)

    clean_input = clean_input.transpose(0,1)
    sim_t = torch.mm(noise_input,clean_input)
    sim_norm = torch.mm(noise_input_norm,clean_input_norm)
    cos_sim = sim_t/sim_norm.clamp(min=eps)
    return cos_sim

def SIM_SELE(index,value):
    top = index.topk(k=1, dim=1, largest=True, sorted=True)
    value = torch.gather(value,1,top[1])
    return torch.where(top[0]/value<1,top[0]/value,value/top[0])
def EuclideanDistances(b,a):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(torch.abs(sum_sq_a+sum_sq_b-2*a.mm(bt)).clamp(min=1e-04))

class SGRAF(object):
    """
    Similarity Reasoning and Filtration (SGRAF) Network
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(
            opt.img_dim, opt.embed_size, no_imgnorm=opt.no_imgnorm
        )
        self.txt_enc = EncoderText(
            opt.vocab_size,
            opt.word_dim,
            opt.embed_size,
            opt.num_layers,
            use_bi_gru=opt.bi_gru,
            no_txtnorm=opt.no_txtnorm,
        )
        self.sim_enc = EncoderSimilarity(
            opt.embed_size, opt.sim_dim, opt.module_name, opt.sgr_step
        )

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,warmup_rate=opt.warmup_rate)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        self.noise_train =  opt.noise_train
        self.noise_tem =  opt.noise_tem

    def state_dict(self):
        state_dict = [
            self.img_enc.state_dict(),
            self.txt_enc.state_dict(),
            self.sim_enc.state_dict(),
        ]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.sim_enc.load_state_dict(state_dict[2])

    def train_start(self):
        """switch to train mode"""
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_enc.train()

    def val_start(self):
        """switch to evaluate mode"""
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_enc.eval()

    def forward_emb(self, images, captions, lengths):
        """Compute the image and caption embeddings"""
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.txt_enc(captions, lengths)
        return img_embs, cap_embs, lengths

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        # Forward similarity encoding
        sims = self.sim_enc(img_embs, cap_embs, cap_lens)
        return sims
    def glo_emb(self,img_emb, cap_emb, cap_lens):
        cap_emb_all = []
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)

        # get enhanced global images by self-attention
        img_ave = torch.mean(img_emb, 1)
        #img_glo = self.v_global_w(img_emb, img_ave) #(batch_size, 1024)
        img_glo = img_ave

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)

            # get enhanced global i-th text by self-attention
            cap_ave_i = torch.mean(cap_i, 1)
            #cap_glo_i = self.t_global_w(cap_i, cap_ave_i)  #(batch_size, 1024)
            cap_emb_all.append(cap_ave_i)

        # (n_image, n_caption)
        cap_emb_all = torch.cat(cap_emb_all, 0)

        return img_glo,cap_emb_all
    def train(
        self,
        images,
        captions,
        lengths,
        hard_negative=True,
        labels=None,
        soft_margin=None,
        mode="train",
        sim_type='euc',
        ids='non',
    ):
        """One epoch training.
        """
        self.Eiters += 1

        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)
        

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.criterion(
            sims,
            hard_negative=hard_negative,
            labels=labels,
            soft_margin=soft_margin,
            mode=mode,
            noise_tem = self.noise_tem
        )

        # return per-sample loss
        if mode == "eval_loss":
            return loss
        if mode =="y_score":
            img_embs,cap_embs = self.glo_emb(images, captions, lengths)
            loss = loss.reshape(-1, 1)
            gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-2, reg_covar=5e-4)
            gmm_A.fit(loss.cpu().numpy())
            #print(y_loss)
            #data = loss.cpu().numpy()
            prob_A = gmm_A.predict_proba(loss.cpu().numpy())
            #class_pre = gmm_A.predict(loss.cpu().numpy())
            prob_A = prob_A[:, gmm_A.means_.argmin()]
            class_c = prob_A>0.8
            class_n = 1-class_c
            if class_c.sum()==img_embs.size()[0]:
                y_value = torch.ones(img_embs.size()[0]).cuda()
                return y_value
            if class_n.sum()==img_embs.size()[0]:
                y_value = torch.zeros(img_embs.size()[0]).cuda()
                return y_value
            clean_index = class_c.nonzero()[0]

            noise_index = class_n.nonzero()[0]
            image_c = img_embs.index_select(0,torch.from_numpy(clean_index).cuda())
            image_n = img_embs.index_select(0,torch.from_numpy(noise_index).cuda())

            text_c = cap_embs.index_select(0,torch.from_numpy(clean_index).cuda())
            text_n = cap_embs.index_select(0,torch.from_numpy(noise_index).cuda())
            if sim_type=='euc':
                img_e_dis = EuclideanDistances(image_c,image_n)
                text_e_dis = EuclideanDistances(text_c,text_n)
                top_img = img_e_dis.topk(k=1, dim=1, largest=False, sorted=True)
                top_text = text_e_dis.topk(k=1, dim=1, largest=False, sorted=True)
                img2text = torch.gather(text_e_dis,1,top_img[1]).float()
                text2img = torch.gather(img_e_dis,1,top_text[1]).float()
                y_half_img = torch.where(top_img[0]/img2text<1,top_img[0]/img2text,img2text/top_img[0])
                y_half_text = torch.where(top_text[0]/text2img<1,top_text[0]/text2img,text2img/top_text[0])
                dis_f_n = (y_half_img+y_half_text)/2
            else:
                sim_img = SIM_PAIR(image_c,image_n)
                sim_text = SIM_PAIR(text_c,text_n)

                img2text = SIM_SELE(sim_img,sim_text)
                text2img = SIM_SELE(sim_text,sim_img)

                dis_f_n = (img2text+text2img)/2
                #dis_f_n =1- (dis_fin - dis_fin.min()) / (dis_fin.max() - dis_fin.min())
                
            y_value = torch.zeros(img_embs.size()[0]).cuda()
                    
            y_value[clean_index]=1
            y_value.scatter_(0, torch.from_numpy(noise_index).cuda(), dis_f_n.clamp(0,1).squeeze(1))
            
            return y_value
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

        return loss.item()
    def sim_score(self,images, captions, lengths,type='sim'):
        if type=='sim':
            img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
            sims = self.forward_sim(img_embs, cap_embs, cap_lens)

            I = self.criterion(sims, mode="predict")
            p = I.clamp(0, 1)
        elif type=='sele':
            img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
            sims = self.forward_sim(img_embs, cap_embs, cap_lens)
            diagonal = sims.diag().view(sims.size(0), 1)
            clean_num = int(sims.size(0)/4)
            top = diagonal.topk(k=clean_num, dim=0, largest=False, sorted=True)
        return p
    def predict(self, images, captions, lengths,images_n='', captions_n='', lengths_n='',epoch=0):
        """
        predict the given samples
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        img_g,text_g = self.sim_enc.glo_emb(img_embs, cap_embs, cap_lens)
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        clean_index,noise_index = self.criterion(sims, mode="predict_clean")
        image_c = img_g[clean_index]
        text_c = text_g[clean_index]
        sim_img = SIM_PAIR(image_c,img_g)
        sim_text = SIM_PAIR(text_c,text_g)

        img2text = SIM_SELE(sim_img,sim_text)
        text2img = SIM_SELE(sim_text,sim_img)

        dis_f_n = 0.5 + (img2text+text2img)/4
        if self.noise_train=='noise_soft':
            index = dis_f_n>self.noise_tem
            dis_f_n = dis_f_n*index
        y_value = torch.zeros(images.size()[0]).cuda()
                
        y_value[clean_index]=1
        y_value.scatter_(0, noise_index, dis_f_n.clamp(0,1).squeeze(1))
        c_y = y_value.clamp(0, 1)

        if epoch:
            img_embs, cap_embs, cap_lens = self.forward_emb(images_n, captions_n, lengths_n)
            img_g_n,text_g_n = self.sim_enc.glo_emb(img_embs, cap_embs, cap_lens)


            sim_img = SIM_PAIR(image_c,img_g_n)
            sim_text = SIM_PAIR(text_c,text_g_n)

            img2text = SIM_SELE(sim_img,sim_text)
            text2img = SIM_SELE(sim_text,sim_img)

            dis_f_n = (img2text+text2img)/2
            if self.noise_train=='noise_soft':
                index = dis_f_n>self.noise_tem
                dis_f_n = dis_f_n*index
            n_y = dis_f_n.clamp(0, 1).squeeze(1)
        else:
            n_y = ''
        return c_y,n_y
