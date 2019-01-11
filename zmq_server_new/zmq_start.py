#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

import argparse
import sys
from zmq_server_new.server import BertServer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_worker', type=int, default=2,
                        help='number of server instances')
    parser.add_argument('-max_batch_size', type=int, default=256,
                        help='maximum number of sequences handled by each worker')
    parser.add_argument('-priority_batch_size', type=int, default=32,
                        help='priority_batch_size')
    parser.add_argument('-port', '-port_in', '-port_data', type=int, default=5555,
                        help='server port for receiving data from client')
    parser.add_argument('-port_out', '-port_result', type=int, default=5556,
                        help='server port for outputting result to client')
    parser.add_argument('-pooling_layer', type=int, nargs='+', default=[-2],
                        help='the encoder layer(s) that receives pooling. '
                             'Give a list in order to concatenate several layers into 1.')
    parser.add_argument('-cpu', type=int, nargs='+', default=True,
                        help='the encoder layer(s) that receives pooling. '
                             'Give a list in order to concatenate several layers into 1.')
    parser.add_argument('-gpu_memory_fraction', type=float, default=0.5,
                        help='determines the fraction of the overall amount of memory '
                             'that each visible GPU should be allocated per worker. '
                             'Should be in range [0.0, 1.0]')
    args = parser.parse_args()
    param_str = '\n'.join(['%20s = %s' % (k, v) for k, v in sorted(vars(args).items())])
    print('usage: %s\n%20s   %s\n%s\n%s\n' % (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))
    return args


if __name__ == '__main__':
    args = get_args()
    server = BertServer(args)
    server.start()
    server.join()
