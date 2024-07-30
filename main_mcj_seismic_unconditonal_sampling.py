import argparse
import traceback
import time
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb
import copy
from runners.ncsn_runner_ncsnv2_seismic import *

import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='marmousi.yml',  help='Path to the config file') #celeba.yml
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--exp_my', type=str, default='D:\\', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='marmousi_1', help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.') #celeba
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--test', default=False, help='Whether to test the model')
    parser.add_argument('--sample', default=False, help='Whether to produce samples from the model')
    parser.add_argument('--fast_fid', default=False, help='Whether to do fast fid test')
    parser.add_argument('--resume_training', default=False, help='Whether to resume training')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    if not args.test and not args.sample and not args.fast_fid:
        if not args.resume_training:
            # if os.path.exists(args.log_path):
            #     overwrite = False
            #     if args.ni:
            #         overwrite = True
            #     else:
            #         response = input("Folder already exists. Overwrite? (Y/N)")
            #         if response.upper() == 'Y':
            #             overwrite = True
            #
            #     if overwrite:
            #         shutil.rmtree(args.log_path)
            #         shutil.rmtree(tb_path)
            #         os.makedirs(args.log_path)
            #         if os.path.exists(tb_path):
            #             shutil.rmtree(tb_path)
            #     else:
            #         print("Folder exists. Program halted.")
            #         sys.exit(0)
            # else:
            #     os.makedirs(args.log_path)

            with open(os.path.join(args.log_path, 'config.yml'), 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)

        new_config.tb_logger = tb.SummaryWriter(log_dir=tb_path)
        # setup logger
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        handler2.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.addHandler(handler2)
        logger.setLevel(level)

    else:
        level = getattr(logging, args.verbose.upper(), None)
        if not isinstance(level, int):
            raise ValueError('level {} not supported'.format(args.verbose))

        handler1 = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler1.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler1)
        logger.setLevel(level)

        if args.sample:
            os.makedirs(os.path.join(args.exp, 'image_samples_mms2'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'image_samples_mms2', args.image_folder)
            # if not os.path.exists(args.image_folder):
            #     os.makedirs(args.image_folder)
            # else:
            #     overwrite = False
            #     if args.ni:
            #         overwrite = True
            #     else:
            #         response = input("Image folder already exists. Overwrite? (Y/N)")
            #         if response.upper() == 'Y':
            #             overwrite = True
            #
            #     if overwrite:
            #         shutil.rmtree(args.image_folder)
            #         os.makedirs(args.image_folder)
            #     else:
            #         print("Output image folder exists. Program halted.")
            #         sys.exit(0)

        elif args.fast_fid:
            os.makedirs(os.path.join(args.exp, 'fid_samples'), exist_ok=True)
            args.image_folder = os.path.join(args.exp, 'fid_samples', args.image_folder)
            if not os.path.exists(args.image_folder):
                os.makedirs(args.image_folder)
            else:
                overwrite = False
                if args.ni:
                    overwrite = False
                else:
                    response = input("Image folder already exists. \n "
                                     "Type Y to delete and start from an empty folder?\n"
                                     "Type N to overwrite existing folders (Y/N)")
                    if response.upper() == 'Y':
                        overwrite = True

                if overwrite:
                    shutil.rmtree(args.image_folder)
                    os.makedirs(args.image_folder)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    args, config = parse_args_and_config()
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")
    print(">" * 80)
    config_dict = copy.copy(vars(config))
    if not args.test and not args.sample and not args.fast_fid:
        del config_dict['tb_logger']
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    # args.resume_training=True
    args.sample=True
    config.sampling.data_init=False
    config.data.seis_rescaled=False

    config.model.sigma_begin=27 #90 9 13 23
    config.model.sigma_end=0.01
    config.model.num_classes=500  #450  500
    config.sampling.batch_size=5 #5
    config.sampling.ckpt_id=210000     #150000
    config.sampling.inpainting = False
    config.sampling.interpolation= False
    config.sampling.n_interpolations=10
    # args.doc = 'marmousi_2'
    # args.log_path = os.path.join(args.exp, 'logs', args.doc)
    # args.image_folder = os.path.join(args.exp, 'image_samples', args.doc)

    args.doc='marmousi_v2_nm'  #marmousi_v2_nm
    args.log_path = os.path.join(args.exp, 'logs', args.doc)
    args.log_path_model ='/home/shendi_mcj/code/Reproducible/ncsnv2-master/exp/logs/marmousi_v2_nm'
    args.image_folder = os.path.join(args.exp, 'image_samples', args.doc)


    try:
        runner = NCSNRunner(args, config)
        runner.sample()
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
