"""Evaluation"""

from __future__ import print_function
import os
import sys
import time
import json
from itertools import chain

import torch
import numpy as np

from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from collections import OrderedDict
from utils import AverageMeter, ProgressMeter
from data import get_dataset, get_loader
import argparse


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(data_loader), [batch_time, data_time], prefix="Encode")

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    # max text length
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    image_ids = []
    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        data_time.update(time.time() - end)
        # image_ids.extend(img_ids)
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            img_embs = np.zeros(
                (len(data_loader.dataset), img_emb.size(1), img_emb.size(2))
            )
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, : max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            progress.display(i)

        del images, captions
    # return img_embs, cap_embs, cap_lens, image_ids
    return img_embs, cap_embs, cap_lens

#def validation_dul(opt, val_loader, models, fold=False):
def validation_dul(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    # compute the encoding for all the validation images and captions
    checkpoint0 = torch.load(model_path[0])
    checkpoint1 = torch.load(model_path[1])
    opt0 = checkpoint0['opt']
    opt1 = checkpoint1['opt']
    model0 = DECL(opt0)
    model1 = DECL(opt1)
    if opt.data_name == 'cc152k_precomp':
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        print(f"No dataset")
        return 0

    models[0].val_start()
    models[1].val_start()
    print('Encoding with model')
    img_embs, cap_embs, cap_lens = encode_data(models[0].similarity_model, val_loader, opt.log_step)
    img_embs1, cap_embs1, cap_lens1 = encode_data(models[1].similarity_model, val_loader, opt.log_step)
    if not fold:
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        img_embs1 = np.array([img_embs1[i] for i in range(0, len(img_embs1), per_captions)])
        # record computation time of validation
        print('Computing similarity from model')
        sims_mean = shard_attn_scores(models[0].similarity_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000,
                                      mode="not sim")
        sims_mean += shard_attn_scores(models[1].similarity_model, img_embs1, cap_embs1, cap_lens1, opt,
                                       shard_size=1000, mode="not sim")
        sims_mean /= 2
        print("Calculate similarity time with model")
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
        print("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
        print("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
        r_sum = r1 + r5 + r10 + r1i + r5i + r10i
        print("Sum of Recall: %.2f" % (r_sum))
        return r_sum
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

            img_embs_shard1 = img_embs1[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard1 = cap_embs1[i * 5000:(i + 1) * 5000]
            cap_lens_shard1 = cap_lens1[i * 5000:(i + 1) * 5000]
            sims = shard_attn_scores(models[0].similarity_model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt,
                                     shard_size=1000,
                                     mode="not sim")
            sims += shard_attn_scores(models[1].similarity_model, img_embs_shard1, cap_embs_shard1, cap_lens_shard1,
                                      opt,
                                      shard_size=1000,
                                      mode="not sim")
            sims /= 2

            print('Computing similarity from model')
            r, rt0 = i2t(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            ri, rti0 = t2i(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        a = np.array(mean_metrics)

        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
        print("rsum: %.1f" % (a[0:3].sum() + a[5:8].sum()))

def evalrank(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint["opt"]
    print("training epoch: ", checkpoint["epoch"])
    opt.workers = 0
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # Load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    if opt.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt.data_path, opt.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader = get_loader(captions, images, split, opt.batch_size, opt.workers)

    # construct model
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    # load model state
    model_A.load_state_dict(checkpoint["model_A"])
    model_B.load_state_dict(checkpoint["model_B"])

    print("Computing results...")
    with torch.no_grad():
        img_embs_A, cap_embs_A, cap_lens_A = encode_data(model_A, data_loader)
        img_embs_B, cap_embs_B, cap_lens_B = encode_data(model_B, data_loader)

    print(
        "Images: %d, Captions: %d"
        % (img_embs_A.shape[0] / per_captions, cap_embs_A.shape[0])
    )

    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs_A = np.array(
            [img_embs_A[i] for i in range(0, len(img_embs_A), per_captions)]
        )
        img_embs_B = np.array(
            [img_embs_B[i] for i in range(0, len(img_embs_B), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        sims_A = shard_attn_scores(
            model_A, img_embs_A, cap_embs_A, cap_lens_A, opt, shard_size=1000
        )
        sims_B = shard_attn_scores(
            model_B, img_embs_B, cap_embs_B, cap_lens_B, opt, shard_size=1000
        )
        sims = (sims_A + sims_B) / 2
        end = time.time()
        print("calculate similarity time:", end - start)

        # bi-directional retrieval
        r, rt = i2t(img_embs_A.shape[0], sims, per_captions, return_ranks=True)
        ri, rti = t2i(img_embs_A.shape[0], sims, per_captions, return_ranks=True)

        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard_A = img_embs_A[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_A = cap_embs_A[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_A = cap_lens_A[i * 5000 : (i + 1) * 5000]

            img_embs_shard_B = img_embs_B[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_B = cap_embs_B[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_B = cap_lens_B[i * 5000 : (i + 1) * 5000]

            start = time.time()
            sims_A = shard_attn_scores(
                model_A,
                img_embs_shard_A,
                cap_embs_shard_A,
                cap_lens_shard_A,
                opt,
                shard_size=1000,
            )
            sims_B = shard_attn_scores(
                model_B,
                img_embs_shard_B,
                cap_embs_shard_B,
                cap_lens_shard_B,
                opt,
                shard_size=1000,
            )
            sims = (sims_A + sims_B) / 2
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])

def evalrank2(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

    # load model and options
    checkpoint = torch.load(model_path[0])
    checkpoint1 = torch.load(model_path[1])
    opt = checkpoint["opt"]
    opt1 = checkpoint1["opt"]
    print("training epoch: ", checkpoint["epoch"])
    opt.workers = 0
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # Load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    if opt.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt.data_path, opt.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader = get_loader(captions, images, split, opt.batch_size, opt.workers)

    # construct model
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    model_A1 = SGRAF(opt1)
    model_B1 = SGRAF(opt1)

    # load model state
    model_A.load_state_dict(checkpoint["model_A"])
    model_B.load_state_dict(checkpoint["model_B"])
    model_A1.load_state_dict(checkpoint1["model_A"])
    model_B1.load_state_dict(checkpoint1["model_B"])

    print("Computing results...")
    with torch.no_grad():
        img_embs_A, cap_embs_A, cap_lens_A = encode_data(model_A, data_loader)
        img_embs_B, cap_embs_B, cap_lens_B = encode_data(model_B, data_loader)
        img_embs_A1, cap_embs_A1, cap_lens_A1 = encode_data(model_A1, data_loader)
        img_embs_B1, cap_embs_B1, cap_lens_B1 = encode_data(model_B1, data_loader)

    print(
        "Images: %d, Captions: %d"
        % (img_embs_A.shape[0] / per_captions, cap_embs_A.shape[0])
    )
    print("1")
    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs_A = np.array(
            [img_embs_A[i] for i in range(0, len(img_embs_A), per_captions)]
        )
        print("2")
        img_embs_B = np.array(
            [img_embs_B[i] for i in range(0, len(img_embs_B), per_captions)]
        )
        print("2")
        img_embs_A1 = np.array(
            [img_embs_A1[i] for i in range(0, len(img_embs_A1), per_captions)]
        )
        print("2")
        img_embs_B1 = np.array(
            [img_embs_B1[i] for i in range(0, len(img_embs_B1), per_captions)]
        )
        print("2")
        # record computation time of validation
        start = time.time()
        sims_A = shard_attn_scores(
            model_A, img_embs_A, cap_embs_A, cap_lens_A, opt, shard_size=1000
        )
        sims_B = shard_attn_scores(
            model_B, img_embs_B, cap_embs_B, cap_lens_B, opt, shard_size=1000
        )
        sims = (sims_A + sims_B) / 2
        print("3")
        sims_A1 = shard_attn_scores(
            model_A1, img_embs_A1, cap_embs_A1, cap_lens_A1, opt1, shard_size=1000
        )
        print("3")
        sims_B1 = shard_attn_scores(
            model_B1, img_embs_B1, cap_embs_B1, cap_lens_B1, opt1, shard_size=1000
        )
        print("3")
        sims0 = (sims_A + sims_B) / 2
        sims1 = (sims_A1 + sims_B1) / 2
        sims = (sims0 + sims1) / 2
        end = time.time()
        print("calculate similarity time:", end - start)

        # bi-directional retrieval
        r, rt = i2t(img_embs_A.shape[0], sims, per_captions, return_ranks=True)
        ri, rti = t2i(img_embs_A.shape[0], sims, per_captions, return_ranks=True)

        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard_A = img_embs_A[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_A = cap_embs_A[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_A = cap_lens_A[i * 5000 : (i + 1) * 5000]

            img_embs_shard_B = img_embs_B[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_B = cap_embs_B[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_B = cap_lens_B[i * 5000 : (i + 1) * 5000]

            img_embs_shard_A1 = img_embs_A1[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_A1 = cap_embs_A1[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_A1 = cap_lens_A1[i * 5000 : (i + 1) * 5000]

            img_embs_shard_B1 = img_embs_B1[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_B1 = cap_embs_B1[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_B1 = cap_lens_B1[i * 5000 : (i + 1) * 5000]

            start = time.time()
            sims_A = shard_attn_scores(
                model_A,
                img_embs_shard_A,
                cap_embs_shard_A,
                cap_lens_shard_A,
                opt,
                shard_size=1000,
            )
            sims_B = shard_attn_scores(
                model_B,
                img_embs_shard_B,
                cap_embs_shard_B,
                cap_lens_shard_B,
                opt,
                shard_size=1000,
            )
            sims0 = (sims_A + sims_B) / 2
            sims_A1 = shard_attn_scores(
                model_A1,
                img_embs_shard_A1,
                cap_embs_shard_A1,
                cap_lens_shard_A1,
                opt1,
                shard_size=1000,
            )
            sims_B1 = shard_attn_scores(
                model_B1,
                img_embs_shard_B1,
                cap_embs_shard_B1,
                cap_lens_shard_B1,
                opt1,
                shard_size=1000,
            )
            sims1 = (sims_A1 + sims_B1) / 2
            sims = (sims0 + sims1) / 2
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])

def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=1000):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    return sims


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    parser.add_argument(
        "--mode_path", default="/data", help="path to datasets"
    )
    parser.add_argument(
        "--mode_path1", default="", help="path to datasets"
    )
    opt = parser.parse_args()
    model_path = opt.mode_path
    model_path1 = opt.mode_path1
    data_path = "NCR-data/data"
    vocab_path = "NCR-data/vocab"
    print(f"loading {model_path}")
    if model_path1:
        evalrank2(
            [model_path,model_path1],
            data_path=data_path,
            vocab_path=vocab_path,
            split="testall",
            fold5=True,
        )
    else:
        evalrank(
            model_path,
            data_path=data_path,
            vocab_path=vocab_path,
            split="testall",
            fold5=True,
        )
