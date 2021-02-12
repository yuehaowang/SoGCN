import subprocess
import os
import sys
import argparse
import time
import subprocess
from functools import partial
import re


int2 = partial(int, base=2)

TASKS_FULLNAME = {
    'molecules': 'molecules_graph_regression',
    'superpixels': 'superpixels_graph_classification',
    'SGS': 'SGS_node_regression',
    'SBMs': 'SBMs_node_classification'
}

SEEDS_LIST = [41, 95, 12, 35]


def ask_y_or_n(q):
    sure = None
    while sure not in ['y', 'n']:
        sure = input('%s (y/n)? ' % (q))
        sure = sure.strip()

    return True if sure == 'y' else False


def main():
    parser = argparse.ArgumentParser(prog='stop_exp.py')
    parser.add_argument('-a', required=True, help='start/stop experiments')
    parser.add_argument('-e', help='experiment ID')
    parser.add_argument('-g', type=int2, help='gpu indices (4 bits binary, e.g. 1011 for enabling GPU0, GPU1 and GPU3)')
    parser.add_argument('-t', required=True, help='task name (molecules/superpixels)') 
    parser.add_argument('-o', default='_out/', help='base directory of outputs')
    parser.add_argument('-r', type=int, default=1, help='the number of trials')
    parser.add_argument('-d', action='store_true', help='keep sessions with a finished experiment')
    parser.add_argument('-v', action='store_true', help='view parameters')
    args, extras = parser.parse_known_args()

    if not args.t in TASKS_FULLNAME:
        print('Task name not found.')
        return

    task_fullname = TASKS_FULLNAME[args.t]

    if args.g is None:
        args.g = 0b1111
    
    if args.e is None:
        args.e = 'tune_' + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')

    gpu_slots = [i for i in range(0, 4) if 1 << i & args.g]
    
    out_path = os.path.join(args.o, task_fullname, args.e) + '/'
    
    print('== Out Dir:', out_path)
    print('== Task:', task_fullname)
    print('== #Trials:', args.r)
    print('== GPU IDs:', gpu_slots)
    print('== Verbose mode: ', args.v)
    print('== Keep finished sessions: ', args.d)
    print('== Extras:', extras)

    if args.a == 'start':
        print('Starting experiments...')

        if os.path.exists(out_path):
            if not ask_y_or_n('The output directory is existing. Proceeding'):
                return

        prefix_cmds = []
        postfix_cmds = []
        if args.d == True:
            prefix_cmds = ['/bin/sh', '-c', '"']
            postfix_cmds = [';', 'exec', 'bash', '"']

        if args.v == True:
            print('Viewing parameters in verbose mode...')
            v_cmd_list = ['python', 'main_%s.py' % task_fullname, '--gpu_id', str(gpu_slots[0]), '--verbose', 'True', '--only_view_params', 'True'] + extras
            subprocess.run(' '.join(v_cmd_list), shell=True)

            if not ask_y_or_n('Continue'):
                return

        idx = 0
        for _ in range(args.r):
            for seed in SEEDS_LIST:
                gpu_id = gpu_slots[idx % len(gpu_slots)]
                cmd_list = ['screen', '-dmS', args.e + '-%s_%s' % (args.t, idx)] + prefix_cmds \
                            + ['python', 'main_%s.py' % task_fullname, '--seed', str(seed), '--max_time', '2048', '--gpu_id', str(gpu_id), '--out_dir', out_path] \
                            + extras + postfix_cmds
                             
                cmd_str = ' '.join(cmd_list)

                subprocess.run(cmd_str, shell=True)
                print(cmd_str)

                idx += 1
                
                time.sleep(4)
    else:
        ps = subprocess.run(['screen', '-ls'], capture_output=True, text=True)
        print(args.e + '-%s' % args.t)
        sessions = [s for s in list(re.findall('\d+\.[\w\.]+-\w+', ps.stdout)) if s.find(args.e + '-%s' % args.t) >= 0]

        print('Sessions will be stopped:')
        if len(sessions) == 0:
            print('\t', '(empty)')
        else:
            for i, s in enumerate(sessions):
                print('\t', '[%d]' % i, s)

            if ask_y_or_n('Proceeding to all'):
                for i, s in enumerate(sessions):
                    subprocess.run(['screen', '-X', '-S', s, 'quit'])
                    print('[%d] %s stopeed' % (i, s))
            else:
                stop_ids = None
                while stop_ids is None:
                    try:
                        stop_ids_str = input('Sessions to be stopped (e.g. "1 2 3"): ')
                        stop_ids = [int(s) for s in stop_ids_str.split()]
                    except Exception:
                        stop_ids = None
                        print('Incorrect input. Session indices should be digits and split by spaces')

                if len(stop_ids) > 0:
                    for s_id in stop_ids:
                        if s_id < len(sessions):
                            subprocess.run(['screen', '-X', '-S', sessions[s_id], 'quit'])
                            print('[%d] %s stopeed' % (s_id, sessions[s_id]))


if __name__ == '__main__':
    main()