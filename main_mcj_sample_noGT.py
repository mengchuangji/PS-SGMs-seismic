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
# from runners import *
from runners.ncsn_runner_mcj_noGT import *
import os

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='marmousi.yml',  help='Path to the config file') #celeba.yml
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--doc', type=str, default='marmousi', help='A string for documentation purpose. '
                                                               'Will be the name of the log folder.') #celeba
    parser.add_argument('--comment', type=str, default='', help='A string for experiment comment')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples") #celeba
    #D:\datasets\CelebA\CelebA\Img\img_align_celeba
    parser.add_argument('-n', '--num_variations', type=int, default=1, help='Number of variations to produce')
    parser.add_argument('-s', '--sigma_0', type=float, default=0.1, help='Noise std to add to observation')
    parser.add_argument('--degradation', type=str, default='den', help='Degradation: inp | deblur_uni | deblur_gauss | sr2 | sr4 | cs4 | cs8 | cs16')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, 'logs', args.doc)


    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f,Loader = yaml.FullLoader)
    new_config = dict2namespace(config)

    tb_path = os.path.join(args.exp, 'tensorboard', args.doc)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    # os.makedirs(os.path.join(args.exp, 'image_samples'), exist_ok=True)
    # args.image_folder = os.path.join(args.exp, 'image_samples', args.image_folder)

    # args.image_folder ='exp/logs/celeba/results/test' # syn | den | inp | cs | deblur_gauss
    args.image_folder = 'exp/logs/marmousi/results'  # syn | den | inp | cs | deblur_gauss

    # if not os.path.exists(args.image_folder):
    #     os.makedirs(args.image_folder)
    # else:
    #     response = input("Image folder already exists. Overwrite? (Y/N)")
    #     if response.upper() == 'Y':
    #         overwrite = True
    #
    #     if overwrite:
    #         shutil.rmtree(args.image_folder)
    #         os.makedirs(args.image_folder)
    #     else:
    #         print("Output image folder exists. Program halted.")
    #         sys.exit(0)

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
    import torch
    # torch.cuda.empty_cache()
    args, config = parse_args_and_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    logging.info("Writing log file to {}".format(args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(args.comment))
    logging.info("Config =")

    args.doc='marmousi_v2_nm'
    args.log_path = os.path.join(args.exp, 'logs', args.doc)
    # Mms35Segyopen/31/500/noNM  Mms35Segyopen_nm/64/550/NM
    # MmsSegyopen_nm/60/550/NM MmsSegyopen/30/500/noNM
    # marmousi_v2_nm/27/500/NM/180000
    args.log_path_model ='/home/shendi_mcj/code/Reproducible/ncsnv2-master/exp/logs/marmousi_v2_nm'
    args.image_folder = 'exp/logs/marmousi_v2_nm/results'

    config.data.image_size=128
    config.sampling.ckpt_id=210000   #v2/13/0.01/450/150000 2/90/0.01/500/180000

    args.degradation='inp' # inp  den
    # args.sigma_0 = 0.2 #denoise 0.203
    config.sampling.batch_size=1
    args.num_variations=3

    config.data.seis_rescaled = False # False True
    config.model.num_classes=500
    # config.model.sigma_begin = 90
    config.model.sigma_begin=27 # 9 13 27
    config.model.sigma_end = 0.01 #0.01
    config.model.sigma_dist = 'geometric'

    print(">" * 80)
    config_dict = copy.copy(vars(config))
    print(yaml.dump(config_dict, default_flow_style=False))
    print("<" * 80)

    #### load data
    import scipy.io as sio
    original = sio.loadmat('/home/shendi_mcj/datasets/seismic/marmousi/marmousi35/marmousi35.mat')[
        'data']  # shape(2441, 13601)
    obs_GT1 = original[750:750 + 128, 7200:7200 + 128]
    plot(obs_GT1, dpi=300, figsize=(3, 3))
    obs_GT = torch.from_numpy(obs_GT1).contiguous().view(1, -1, obs_GT1.shape[-2], obs_GT1.shape[-1]).type(
        torch.FloatTensor)

    # The noise level you set when testing
    sigma_preset=0.8*abs(observation_['H']).max()
    print("sigma_preset:", sigma_preset)

    obs=obs_GT+sigma_preset*torch.randn_like(obs_GT)

    # Automatic noise level estimation by VI-non-IID
    from utils.estimate_sigma_using_VInonIID import estimate_sigma_using_VInonIID
    sigma_dict=estimate_sigma_using_VInonIID(obs[0].view(1,-1,obs.shape[2],obs.shape[3]))
    args.sigma_0 = sigma_dict['median']



    try:
        runner = NCSNRunner(args, config)
        runner.sample(obs)
    except:
        logging.error(traceback.format_exc())

    return 0


if __name__ == '__main__':
    sys.exit(main())
