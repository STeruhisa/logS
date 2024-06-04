prediction and analysis in log S
====

〜という論文に使用したコードとデータです。このリポジトリはGNNによる学習モデルの構築、予測、結果の分析、データの4つにカテゴライズされています。
## Description
### Data
原子の特徴はMAPとIAPの2種類の中から選ぶことができ、使用しない方をコメントアウトして使います。<br>(example)もしあなたがIAPを使用したいなら
　#####   MAP
　#   'prop_list': ['csed_charge','fukui_minus','volume','c6','Faniso_shield','atom_orbene','voronoi','afluc'],
　#   'coarse_graining_params': { 'coarse_graining': 0, # performe coarse graining (1) or not (0)
　#                               'num_bin' : 8,        # number of bins for the coarse graining
　#                               'prop_min': [-1.0,-0.43,3.0,0.0,0.0,0.0,0.0,0.0],
　#                               'prop_max': [1.0,0.53,60.4,100.0,1.0,2.0,0.7,1.4]
　#                              }, 
　######   IAP 
     'prop_list': ['eff_chg','pol','radii','ion1','aff','mass','voronoi','afluc'],
     'coarse_graining_params': { 'coarse_graining': 0, # performe coarse graining (1) or not (0)
                                 'num_bin' : 8,        # number of bins for the coarse graining
                                 'prop_min': [1.0,3.0,20.0,10.0,0.0,1.0,0.0,0.0],
                                 'prop_max': [10.0,30.0,120.0,20.0,5.0,100.0,1.0,10.0]
                               },







### Training

### Prediction

### Analysis


## Requirement
- python
- numpy
- pandas
- pytorch_lightning
- torch
- 
## Install

## Usage

## Demo

## Licence

[XXX](https://github.com/XXX)

## Reference
[1]User Name, 'Paper Titile' Conference Name pp.xx 20XX

[XXX](https://github.com/XXX)
