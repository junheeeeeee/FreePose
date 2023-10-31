import argparse



def get_parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # Evaluate choice
    parser.add_argument('--eval_h36m', default=True, type=bool, metavar='NAME', help='h36m dataset')
    parser.add_argument('--eval_h36mp', default=True, type=bool, metavar='NAME', help='h36mp dataset')
    parser.add_argument('--eval_hp3d', default=True, type=bool, metavar='NAME', help='3dhp dataset')
    parser.add_argument('--eval_pw3d', default=True, type=bool, metavar='NAME', help='3dpw dataset')

    # Model arguments
    parser.add_argument('--posenet_name', default='simple', type=str, help='posenet: gcn/stgcn/videopose/mlp')
    # parser.add_argument('--stages', default=4, type=int, metavar='N', help='stages of baseline model')
    # parser.add_argument('--dropout', default=0.25, type=float, help='dropout rate')

    # Training detail
    parser.add_argument('--batch_size', default=1024, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--data_size', default=100, type=int, metavar='N', help='number of training data')
    parser.add_argument('--frame_size', default=9, type=int, metavar='N', help='number of frame')
    # Learning rate
    parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
    # parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    # parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    # parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)

    # Experimental setting
    # parser.add_argument('--random_seed', type=int, default=0)
    # parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')
    # parser.add_argument('--pretrain', default=False, type=lambda x: (str(x).lower() == 'true'), help='used in poseaug')
    # parser.add_argument('--s1only', default=False, type=lambda x: (str(x).lower() == 'true'), help='train S1 only')
    parser.add_argument('--num_workers', default=6, type=int, metavar='N', help='num of workers for data loading')

    args = parser.parse_args()

    return args
