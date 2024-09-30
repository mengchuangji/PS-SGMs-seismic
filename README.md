# Stochastic solutions for simultaneous seismic data denoising and reconstruction via score-based generative models

This repo contains the official implementation for the paper [Stochastic solutions for simultaneous seismic data denoising and reconstruction via score-based generative models
](https://ieeexplore.ieee.org/abstract/document/10579850). 

by Chuangji Meng, Jinghuai Gao∗, Yajun Tian, Hongling Chen∗, Wei Zhang, Renyu Luo.

This is an example of schematic diagram of conditional posterior sampling.
![x_y_evolve](assets/x_y_evolve.jpg)


## Running Experiments

### Dependencies

Run the following conda line to install all necessary python packages for our code and set up the environment.

```bash
conda env create -f environment.yml
```

The environment includes `cudatoolkit=11.0`. You may change that depending on your hardware.

### Project structure

`main.py` is the file that you should run for both training and sampling. Execute ```python main.py --help``` to get its usage description:

```
usage: main.py [-h] --config CONFIG [--seed SEED] [--exp EXP] --doc DOC
               [--comment COMMENT] [--verbose VERBOSE] [-i IMAGE_FOLDER]
               [-n NUM_VARIATIONS] [-s SIGMA_0] [--degradation DEGRADATION]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name
                        of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  -n NUM_VARIATIONS, --num_variations NUM_VARIATIONS
                        Number of variations to produce
  -s SIGMA_0, --sigma_0 SIGMA_0
                        Noise std to add to observation
  --degradation DEGRADATION
                        Degradation: inp | den 
                        

```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── image_samples 
│  
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      └── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
├── image_samples # contains generated samples
│   └── <i>
│       ├── stochastic_variation.png # samples generated from checkpoint_x.pth, including original, degraded, mean, and std   
│       ├── results.mat # the pytorch tensor corresponding to stochastic_variation.png
│       └── y_0.mat # the pytorch tensor containing the input y of SNIPS
```

### data preparation/ Downloading data
You can test on Marmousi (mat file format)/ [Opensegy]("http://s3.amazonaws.com/open.source.geoscience/open_data) (SEGY/SGY file format) / field data (SEGY/SGY file format)
## marmousi
This is an example of schematic diagram of conditional posterior sampling for denoising.
![mms_xt_den_evol](assets/mms_xt_den_evol.png)

This is an example of schematic diagram of conditional posterior sampling for reconstruction.
![mms_xt_cm_evol](assets/mms_xt_cm_evol.png)

## field data
This is an example of schematic diagram of conditional posterior sampling for field data.
![obs_xt_cm_evol](assets/obs_xt_cm_evol.png)

### Running 

If we want to run sampling on marmousi for the problem of reconstruction, with added noise of standard deviation 0.1, and obtain 3 variations, we can run the following

```bash
python ncsn_runner_mcj_GT.py -i images --config marmousi.yml --doc marmousi_v2_nm -n 3 --degradation inp --sigma_0 0.1
```
Samples will be saved in `<exp>/image_samples/marmousi_v2_nm`.

The available degradations are: Denoising (`den`), reconstruction (`inp`). The sigma_0 (noise level of observation) can be set manually or estimated automatically.

If you don't need GT to evaluate the results, use main_mcj_sample_noGT.py for synthetic data (e.g., mat foramt file) and main_mcj_sample_noGT_field.py for real data (SEGY/SGY file format).

for denoising, you should  use general_anneal_Langevin_dynamics_den  (function) in runners/ncsn_runner_mcj_noGT.py/runners/ncsn_runner_mcj_GT.py
for reconstruction, you should  use general_anneal_Langevin_dynamics_inp (function)  in runners/ncsn_runner_mcj_noGT.py

## Pretrained Checkpoints

We provide two trained models and log files in files `<exp>/logs/marmousi_v2_nm` and `<exp>/logs/MmsSegyopenf`. see [pretrained model](https://pan.baidu.com/s/1p5y_JC1AWSD7QCWRsSwMFw?pwd=1111), 提取码: 1111.

These checkpoint files are provided as-is from the authors of [NCSNv2_seismic](https://github.com/mengchuangji/ncsnv2_seismic).

**Note**:
----------------------------------------------------------------------------------------
You can also retrain a SGMs (change hyperparameters or datasets , see [NCSNv2_seismic](https://github.com/mengchuangji/ncsnv2_seismic)) to get your own generative model for seismic data or other types of natural image data, remote sensing data, medical imaging data.

## Acknowledgement

This repo is largely based on the [NCSNv2](https://github.com/ermongroup/ncsnv2) repo and  [NCSNv2](https://github.com/mengchuangji/ncsnv2_seismic) repo, and uses modified code from [SNIPS](https://github.com/bahjat-kawar/snips_torch) for implementation of conditional score function and [VI-non-IID](https://github.com/mengchuangji/VI-Non-IID) for automatic noise level estimation. Thanks for their contributions.

----------------------------------------------------------------------------------------
**Note**: This project is being integrated into commercial software and is intended for scientific research purposes only, not commercial purposes.
-----------------------------------------------------------------------------------------



## References

If you find the code/idea useful for your research, please consider citing

```bib
@ARTICLE{10579850,
  author={Meng, Chuangji and Gao, Jinghuai and Tian, Yajun and Chen, Hongling and Zhang, Wei and Luo, Renyu},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Stochastic Solutions for Simultaneous Seismic Data Denoising and Reconstruction via Score-Based Generative Models}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Noise reduction;Data models;Inverse problems;Noise;Mathematical models;Training;Stochastic processes;Denoising;Langevin dynamics;posterior sampling;reconstruction;score-based generative models (SGMs);stochastic solutions},
  doi={10.1109/TGRS.2024.3421597}}
```
and/or our earlier work on generative modeling of seismic data

```bib
@inproceedings{meng2024generative,
  title={Generative Modeling of Seismic Data Using Score-Based Generative Models},
  author={Meng, C and Gao, J and Tian, Y and Chen, H and Luo, R},
  booktitle={85th EAGE Annual Conference \& Exhibition (including the Workshop Programme)},
  volume={2024},
  number={1},
  pages={1--5},
  year={2024},
  organization={European Association of Geoscientists \& Engineers}
}
```
and/or our earlier work on simultaneous denoising and noise level estimation

```bib
@ARTICLE{9775677,
  author={Meng, Chuangji and Gao, Jinghuai and Tian, Yajun and Wang, Zhiqiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Seismic Random Noise Attenuation Based on Non-IID Pixel-Wise Gaussian Noise Modeling}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  keywords={Attenuation;Noise measurement;Gaussian noise;Data models;Noise reduction;Noise level;Training;Deep learning (DL);noise estimation;noise modeling;non-independently identically distribution (IID);seismic random noise attenuation (NA);variational inference (VI)},
  doi={10.1109/TGRS.2022.3175535}}
```


