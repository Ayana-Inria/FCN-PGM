This repository contains the code related to the paper:  

M. Pastorino, G. Moser, S. B. Serpico, and J. Zerubia, "Semantic Segmentation of Remote-Sensing Images Through Fully Convolutional Neural Networks and Hierarchical Probabilistic Graphical Models," IEEE Transactions on Geoscience and Remote Sensing, 2022, [https://inria.hal.science/hal-03534026v1](https://inria.hal.science/hal-03534026v1)[https://ieeexplore.ieee.org/document/9676578](https://ieeexplore.ieee.org/document/9676578).

When using this work, please cite our IEEE TGRS'22 paper:

```
@ARTICLE{pastorino_tgrs22,
  author={Pastorino, Martina and Moser, Gabriele and Serpico, Sebastiano B. and Zerubia, Josiane},
  journal={IEEE International Conference on Image Processing}, 
  title={Semantic Segmentation of Remote-Sensing Images Through Fully Convolutional Neural Networks and Hierarchical Probabilistic Graphical Models}, 
  year={2022},
  volume={60},
  number={},
  pages={1-16},
  doi={10.1109/TGRS.2022.3141996}}

```

## :carousel_horse: Installation

The code was built on a virtual environment running on Python 3.9

### :clock1: Step 1: Clone the repository

```
git clone --recursive https://github.com/Ayana-Inria/FCN-PGM.git
```

### :clock2: Step 2: Clone the repository

```
cd FCN-PGM

pip install -r requirements.txt
```

### :clock3: Step 3: Run the code

Please refer to the notebook FCN-PGM.ipynb for more explanations.  


## :roller_coaster: Project structure

```
semantic_segmentation
├── dataset - contains the data loader
├── input - images to train and test the network 
├── net - contains the loss, the network, and the training and testing functions
├── output - should contain the results of the training / inference
|   ├── exp_name
|   └── model.pth
├── utils - misc functions
└── main.py - program to run
```
  
## :bento: Data

The model is trained on the [ISPRS Vaihingen dataset](http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-vaihingen.html) and [ISPRS Potsdam dataset](http://www2.isprs.org/potsdam-2d-semantic-labeling.html). The two datasets consist of VHR optical images (spatial resolutions of 9 and 5cm, respectively), we used the IRRG channels. They can be downloaded on [Kaggle](https://www.kaggle.com/datasets/bkfateam/potsdamvaihingen) and should be inserted in the folder `/input`.

The data should have the following structure. 

```
input
├── top
|   └── top_mosaic_09cm_area{}.tif
├── gt
|   └── top_mosaic_09cm_area{}.tif
└── gt_eroded
    └── top_mosaic_09cm_area{}_noBoundary.tif
```


## :new_moon_with_face: License

The code is released under the GPL-3.0-only license. See `LICENSE.md` for more details.

## :wind_face: Acknowledgements

This work was conducted during my joint PhD at [INRIA](https://team.inria.fr/ayana/team-members/), d'Université Côte d'Azur and at the [University of Genoa](http://phd-stiet.diten.unige.it/). 
The ISPRS 2D Semantic Labeling Challenge Datasets were provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF).
The code to deal with the ISPRS dataset derives from the GitHub repository [Deep learning for Earth Observation](https://github.com/nshaud/DeepNetsForEO).
