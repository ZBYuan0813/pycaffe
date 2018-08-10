#coding:utf-8
import argparse
import sys
sys.path.insert(0,'/mnt/zanghao/caffe_video_triplet/caffe_video_triplet/'+'python')
from create_solver import CaffeSolver
from train import train
#from create_proto import write_net
#所有的训练参数都在这里定义，更改对应的参数即可
def main(args):

    #train_proto,test_proto=write_net(args)
    #args.train_net = train_proto
    #args.test_net = test_proto
    args.train_net = '"../df/yi+shopping.prototxt"'
    solver = CaffeSolver(args)
    solver_path = solver.write_solver()
   # solver_path = 'solver/solver1.prototxt'
    # set model and prototxt path
    model_path = '/mnt/zanghao/models/yi+shopping.caffemodel'
    
    train(args,solver_path,model_path)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    # create train and test prototxt parameter
    parser.add_argument('--train_lmdb', type=str,
        help='Directory of train lmdb.',
        default="train.lmdb")
    parser.add_argument('--test_lmdb',type=str,
        help='Directory of test lmdb.',
        default='test.lmdb')
    parser.add_argument('--mean_file',type=str,
        help='Directory of mean file.',
        default='mean.binaryproto')
    parser.add_argument('--train_proto',type=str,
        help='Directory to store train.prototxt',
        default='train.prototxt')
    parser.add_argument('--test_proto',type=str,
        help='Directory to store test.prototxt',
        default='test.prototxt')
    parser.add_argument('--loss_type',type=str,
        help='Loss function you choose.',
        default='SoftmaxWithLoss')


    #####################################################################
    # generator solver parameter
    parser.add_argument('--train_net', type=str, 
        help='Directory of train.prototxt.', 
        default='"train.prototxt"')
    parser.add_argument('--test_net', type=str,
        help='Directory of test.prototxt.', 
        default='')
    parser.add_argument('--test_iter', type=int,
        help='Test iter.', 
        default=500)
    parser.add_argument('--test_interval', type=int,
        help='How long to test.')
    parser.add_argument('--base_lr', type=float,
        help='The learn rate for network.',
        default = 0.01)
    parser.add_argument('--display', type=int,
        help='How many iterators to show the tain information.', 
        default=50)
    parser.add_argument('--max_iter', type=int,
        help='Number of epochs to run.', 
        default=10000)
    parser.add_argument('--lr_policy', type=str,
        help='Gradient optimization method.', 
        default='"step"')
    parser.add_argument('--gamma', type=float,
        help='Learning rate change index.', 
        default=0.1)
    parser.add_argument('--momentum', type=float,
        help='Momentum.', default=0.9)
    parser.add_argument('--weight_decay', type=float,
        help='Weight decay.', 
        default=0.0005)
    parser.add_argument('--stepsize', type=int,
        help='Number of step to change learn rate.', 
        default=3000)
    parser.add_argument('--snapshot', type=int,
        help='Store model in training.', 
        default=10000)
    parser.add_argument('--snapshot_prefix', type=str,
        help='Name of stored model.', 
        default='"snapshot"')
    parser.add_argument('--solver_mode', type=str,
        help='Use GPU' , 
        default='GPU')
    parser.add_argument('--solver_type', type=str,
        help='Solver method.', 
        default='SGD')
    parser.add_argument('--device_id', type=int,
        help='Which gpu to use.',
        default=0)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
