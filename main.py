#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys
import time
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options

from model import LinearModel, weight_init
from train import DatasetTrain, DatasetTest
import util
import log



def main(opt):
    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options

    # create model
    print(">>> creating model")
    model = LinearModel()
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        # import pdb; pdb.set_trace()
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    

    # list of action(s)
   

    # data loading
    # test
    if opt.test:
        
        test_loader = DataLoader(DatasetTest('test.npy','label.npy'), batch_size =128,drop_last=False)
               
        hh=test(test_loader, model, criterion)
           
        print (">>>>>> TEST results:")
       
        sys.exit()

    # load dadasets for training
    test_loader = DataLoader(DatasetTest('train.npy','label.npy'), batch_size=128,drop_last=False)
    train_loader = DataLoader(DatasetTrain('train.npy','label.npy'), batch_size=128,drop_last=False, shuffle=True)
    print(">>> data loaded !")

    cudnn.benchmark = True
    for epoch in range(start_epoch, opt.epochs):
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))

        # per epoch
        glob_step, lr_now, loss_train = train(
            train_loader, model, criterion, optimizer,
            lr_init=opt.lr, lr_now=lr_now, glob_step=glob_step, lr_decay=opt.lr_decay, gamma=opt.lr_gamma,
            max_norm=opt.max_norm)
        loss_test = test(test_loader, model, criterion)

        # update log file
       

        # save ckpt
        is_best = loss_test < err_best
        err_best = min(loss_test, err_best)
        if is_best:
        	log.save_ckpt({'epoch': epoch + 1,
                           'lr': lr_now,
                           'step': glob_step,
                           'err': err_best,
                           'state_dict': model.state_dict(),
                           'optimizer': optimizer.state_dict()},
                          ckpt_path=opt.ckpt,
                          is_best=True)
        

   


def train(train_loader, model, criterion, optimizer,
          lr_init=None, lr_now=None, glob_step=None, lr_decay=None, gamma=None,
          max_norm=True):
    losses = util.AverageMeter()

    model.train()

    start = time.time()
    batch_time = 0
   
    for i, (inps, tars) in enumerate(train_loader):
        glob_step += 1
        if glob_step % lr_decay == 0 or glob_step == 1:
            lr_now = util.lr_decay(optimizer, glob_step, lr_init, lr_decay, gamma)
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda())

        outputs = model(inputs)

        # calculate loss
        optimizer.zero_grad()
        loss = criterion(outputs, targets)
        losses.update(loss.item(), inputs.size(0))
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
    print (">>> train error: {} <<<".format(losses.avg))

        # update summary

       
    return glob_step, lr_now, losses.avg


def test(test_loader, model, criterion):
    losses = util.AverageMeter()

    model.eval()

    all_dist = []
    start = time.time()
    batch_time = 0
    results = list()
    for i, (inps, tars) in enumerate(test_loader):
        inputs = Variable(inps.cuda())
        targets = Variable(tars.cuda())

        
        outputs = model(inputs)
        results.append(outputs)

        # calculate loss
        outputs_coord = outputs
        loss = criterion(outputs_coord, targets)

        losses.update(loss.item(), inputs.size(0))

    final=torch.cat(results,dim=0)
    final=1000000*final.detach().cpu().numpy()[:,0]
    results_txt = open("results.txt", "w+")
    for i in range(final.shape[0]):
    	results_txt.write(str(final[i]) + '\n')
        # update summary
       
    print (">>> error: {} <<<".format(losses.avg))
    return losses.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)

