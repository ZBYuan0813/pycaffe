#coding: utf-8
import os
import sys
sys.path.insert(0,'/mnt/zanghao/caffe_video_triplet/caffe_video_triplet'+'python')
import caffe
import numpy as np 


def train(args,solver_path,model_path):
    print 'training.......'
    niter = 200
    train_loss = np.zeros(niter)
    scratch_train_loss = np.zeros(niter)

    # use gpu
    caffe.set_device(args.device_id)
    caffe.set_mode_gpu()
    #use cpu
    #caffe.set_mode_cpu()
    # We create a solver that fine-tunes from a previously trained network.
    solver = caffe.SGDSolver(solver_path)
    solver.net.copy_from(model_path)
    # For reference, we also create a solver that does no finetuning.
    scratch_solver = caffe.SGDSolver(solver_path)
    # train and test network    
    for it in range(niter):
        solver.step(1)  # SGD by Caffe
        scratch_solver.step(1)
        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data
        scratch_train_loss[it] = scratch_solver.net.blobs['loss'].data
        if it % args.display == 0:
            print 'iter %d, finetune_loss=%f, scratch_loss=%f' % (it, train_loss[it], scratch_train_loss[it])
        if args.test_net != '':
            if it % args.test_iters == 0:
                test_iters = 10
                accuracy = 0
                scratch_accuracy = 0
                for it in arange(test_iters):
                    solver.test_nets[0].forward()
                    accuracy += solver.test_nets[0].blobs['accuracy'].data
                    scratch_solver.test_nets[0].forward()
                    scratch_accuracy += scratch_solver.test_nets[0].blobs['accuracy'].data
                accuracy /= test_iters
                scratch_accuracy /= test_iters
                print 'Accuracy for fine-tuning:', accuracy
                print 'Accuracy for training from scratch:', scratch_accuracy    
