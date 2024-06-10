prediction and analysis in log S
====

〜という論文に使用したコードとデータです。このリポジトリにあるコードを実行することによってgraph nural networkを使った学習モデルの構築と予測を行うことができます。
## Description
### Data
原子の特徴はMAPとIAPの2種類の中から選ぶことができ、使用しない方をコメントアウトして使います。<br>(example)もしあなたがIAPを使用したいならregression.pyに対して

![choice](https://github.com/STeruhisa/logS/assets/171115343/dc9a69f3-04ef-4c17-99fa-28a3c2ddf47d)






### Training
トレーニングを行うにあたり、下図の値はoptuna使用したハイパーパラメータのfittingを行うことが可能です。その場合、./training.sh -hyper というふうに実行します。得られた適切なパラメータを使用したトレーニングはregression.pyを更新します。

![スクリーンショット 2024-06-10 14 54 56](https://github.com/STeruhisa/logS/assets/171115343/caa6daba-5cdb-41ef-a526-a2ebf3bedc6d)


その後、./training.sh を実行すればtraining　は完了で、ここで得られたtraining情報は.ckptファイルに保存されます。私が実行した一部のckptファイルが置いてあります。
### Prediction
./prediction.sh でpredicition.py を実行します。

## Requirement
- python 3.7.13
- numpy 1.21.5
- pandas 1.3.5
- pytorch_lightning 1.9.4
- torch 1.13.1
- tqdm 4.66.1
