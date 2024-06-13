prediction in log S
====

It is code and data used in the paper "Feature attributions for water-solubility predictions obtained by artificial intelligence methods and chemists". The code in this repository can be run to architect graph neural network models and make predictions.

## Description
Please select either "MAPs" or "IAPs" and comment out the one you don't use.<br>(example)If you want to run an IAP, see the figure below against **`regression.py`** 


![choice](https://github.com/STeruhisa/logS/assets/171115343/dc9a69f3-04ef-4c17-99fa-28a3c2ddf47d)






### Training
トレーニングを行うにあたり、下図の値はoptuna使用したハイパーパラメータのfittingを行うことが可能です。その場合、./training.sh -hyper というふうに実行します。得られた適切なパラメータを使用したトレーニングはregression.pyを更新します。

![スクリーンショット 2024-06-10 14 54 56](https://github.com/STeruhisa/logS/assets/171115343/caa6daba-5cdb-41ef-a526-a2ebf3bedc6d)


その後、./training.sh を実行すればtraining　は完了で、ここで得られたtraining情報は.ckptファイルに保存されます。私が実行した一部のckptファイルが置いてあります。
### Prediction
./prediction.sh でpredicition.py を実行します。lavelが実験値、ckpt0-4が予測値としてcsvファイルに保存されます。

## Requirement
- python 3.7.13
- numpy 1.21.5
- pandas 1.3.5
- pytorch_lightning 1.9.4
- torch 1.13.1
- tqdm 4.66.1
