"""Training script"""

import os
import time
import copy
import shutil
import random
import logging
import numpy as np
import torch
import scipy.stats as stats
from sklearn.mixture import GaussianMixture
from evaluation import evalrank
from data import get_loader, get_dataset
from model import SGRAF
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
)

################### CODE FOR THE BETA MODEL  ########################
def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta

class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)

def main(opt):
    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    best_rsum = 0
    start_epoch = 0
    best_test  = 0
    # save the history of losses from two networks
    all_loss = [[], []]

    # Warmup
    print("\n* Warmup")
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model_A.load_state_dict(checkpoint["model_A"])
            model_B.load_state_dict(checkpoint["model_B"])
            print(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.warmup_model_path, checkpoint["epoch"]
                )
            )
            print("\nValidattion ...")
            validate(opt, val_loader, [model_A, model_B])
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    else:
        epoch = 0
        for epoch in range(0, opt.warmup_epoch):
            print("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_A, epoch)
            print("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_B, epoch)

    # save the history of losses from two networks
    all_loss = [[], []]
    print("\n* Co-training")
    
    if opt.saved_model:
        checkpoint = torch.load(opt.saved_model)
        model_A.load_state_dict(checkpoint["model_A"])
        model_B.load_state_dict(checkpoint["model_B"])
        start_epoch = checkpoint["epoch"]+1
    # Train the Model
    row_data = []
    row_data_val = []
    for epoch in range(start_epoch, opt.num_epochs):
        print("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (labeled, unlabeled)
        print("Split dataset ...")
        prob_A, prob_B,all_loss = eval_train(
            opt,
            model_A,
            model_B,
            noisy_trainloader,
            data_size,
            all_loss
        )

        pred_A = split_prob(prob_A, opt.p_threshold)
        pred_B = split_prob(prob_B, opt.p_threshold)

        print("\nModel A training ...")
        # train model_A
        if len(pred_B.nonzero()[0]):
            labeled_trainloader, unlabeled_trainloader = get_loader(
                captions_train,
                images_train,
                "train",
                opt.batch_size,
                opt.workers,
                opt.noise_ratio,
                opt.noise_file,
                pred=pred_B,
                prob=prob_B,
            )
            train(opt, model_A, model_B, labeled_trainloader, unlabeled_trainloader, epoch)

        print("\nModel B training ...")
        # train model_B
        if len(pred_A.nonzero()[0]):
            labeled_trainloader, unlabeled_trainloader = get_loader(
                captions_train,
                images_train,
                "train",
                opt.batch_size,
                opt.workers,
                opt.noise_ratio,
                opt.noise_file,
                pred=pred_A,
                prob=prob_A,
            )
            train(opt, model_B, model_A, labeled_trainloader, unlabeled_trainloader, epoch)

        print("\nValidattion ...")
        # evaluate on validation set
        rsum,result_list = validate(opt, val_loader, [model_A, model_B])
        row_data_val.append(result_list)
        logging.info("epoch")
        logging.info(epoch)
        logging.info("result_list")
        logging.info(result_list)
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_A": model_A.state_dict(),
                "model_B": model_B.state_dict(),
                "best_rsum": best_rsum,
                "opt": opt,
            },
            is_best,
            filename="checkpoint_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )
        

def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader=None, epoch=None):
    """
    One epoch training.
    """
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, losses],
        prefix="Training Step",
    )

    # fix one network and train the other
    net.train_start()
    net2.val_start()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    labels_l = []
    labels_u = []
    end = time.time()
    for i, batch_train_data in enumerate(labeled_trainloader):
        (
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            _,
            batch_labels_l,
            batch_prob_l,
            batch_clean_labels_l,
        ) = batch_train_data
        batch_size = batch_images_l.size(0)
        labels_l.append(batch_clean_labels_l)

        # unlabeled data
        try:
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                _,
                batch_clean_labels_u,
            ) = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                _,
                batch_clean_labels_u,
            ) = unlabeled_train_iter.next()
        labels_u.append(batch_clean_labels_u)

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_prob_l = batch_prob_l.cuda()
            batch_labels_l = batch_labels_l.cuda()

        # label refinement
        
        # drop last batch if only one sample (batch normalization require)
        if batch_images_l.size(0) == 1 or batch_images_u.size(0) == 1:
            break

        net.train_start()
        # train with labeled + unlabeled data  exponential or linear
        
        if epoch < (opt.num_epochs // 2):
            loss_u = 0
            with torch.no_grad():
                net2.val_start()
                c_y,n_y = net2.predict(batch_images_l, batch_text_l, batch_lengths_l)
        else:
            with torch.no_grad():
                net2.val_start()
                c_y,n_y = net2.predict(batch_images_l, batch_text_l, batch_lengths_l,batch_images_u, batch_text_u, batch_lengths_u,epoch=epoch)

            loss_u = net.train(
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                labels=n_y,
                hard_negative=True,
                soft_margin=opt.soft_margin,
                mode=opt.noise_train,
            )
        loss_l = net.train(
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            labels=c_y,
            hard_negative=True,
            soft_margin=opt.soft_margin,
            mode=opt.noise_train,
        )
        loss = loss_l + loss_u
        losses.update(loss, batch_images_l.size(0) + batch_images_u.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss = model.train(images, captions, lengths, mode=opt.warmup_type)
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )
    result_list = [(r1+r5+r10)/3,r1, r5, r10]
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i
    result_list+=[(r1i+r5i+r10i)/3,r1i, r5i, r10i]
    return r_sum,result_list


def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        with torch.no_grad():
            # compute the loss
            loss_A = model_A.train(images, captions, lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)
    print("\nFitting GMM ...")
    # fit a two-component GMM to the loss
    if opt.fit_type == 'gmm':
        gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_A.fit(input_loss_A.cpu().numpy())
        prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
        prob_A = prob_A[:, gmm_A.means_.argmin()]

        gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_B.fit(input_loss_B.cpu().numpy())
        prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
        prob_B = prob_B[:, gmm_B.means_.argmin()]
    else:
        bmm_A = BetaMixture1D(max_iters=10)
        bmm_A.fit(input_loss_A.cpu().numpy())
        prob_A = bmm_A.posterior(input_loss_A.cpu().numpy(),0)

        bmm_B = BetaMixture1D(max_iters=10)
        bmm_B.fit(input_loss_B.cpu().numpy())
        prob_B = bmm_B.posterior(input_loss_B.cpu().numpy(),0)
    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred

def EuclideanDistances(a,b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a,dim=1).unsqueeze(1)  # m->[m, 1]
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b,dim=1).unsqueeze(0)  # n->[1, n]
    bt = b.t()
    return torch.sqrt(sum_sq_a+sum_sq_b-2*a.mm(bt))

def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred
