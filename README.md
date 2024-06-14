prediction in log S
====

This git repogitory contents contains code and data used in the paper "Feature attributions for water-solubility predictions obtained by artificial intelligence methods and chemists". This code in this repository can be run to architect graph neural network models and make predictions.

## Description
Please select either "MAPs" or "IAPs" and comment out the one you don't use.<br>(example) If you want to run IAPs, see the figure below against **`regression.py`** 


![choice](https://github.com/STeruhisa/logS/assets/171115343/dc9a69f3-04ef-4c17-99fa-28a3c2ddf47d)






### Training
In training, the code allows for the fitting of hyperparameters using Optuna. In this case, run **`./training.sh -hyper`**. If you want to use the optimal parameters obtained, update the hyper parameters section in **`regression.py`** (the numerical parts of orange color in the figure below).

![スクリーンショット 2024-06-10 14 54 56](https://github.com/STeruhisa/logS/assets/171115343/caa6daba-5cdb-41ef-a526-a2ebf3bedc6d)

After the parameters are updated, if you operate **`./traing.sh`** and will finish training shortly. The obtained training information is saved in the **`.ckpt`** file. Some of the **`.ckpt`** files were placed in the **`dirckpt`** directory, as examples.

### Prediction
**`./prediction.sh`** to run **`predicition.py`**, where label is the experimental value and ckpt0-4 is the predicted value, saved in a **`.csv`** file.

## Requirement
- python 3.7.13
- numpy 1.21.5
- pandas 1.3.5
- pytorch_lightning 1.9.4
- torch 1.13.1
- tqdm 4.66.1
