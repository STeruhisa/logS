import argparse
import glob
import numpy as np
import pandas as pd
import runpy
import torch
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F

from moduleNN.data.dataloader import SDFDataLoader
from moduleNN.data.dataset import MoleculeDataset, gcn_collate_fn
from moduleNN.models.graph_conv_model import GraphConvModel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-accelerator", default=None)
    args = parser.parse_args()

    global params

    params = runpy.run_path(args.config).get('model_params', None)
    pl.seed_everything(params["random_seed"])

    ckpt_file_list = glob.glob(params["ckpt_dir"]+"/"+params["ckpt_name"]+'_best_loss_*.ckpt')

#   checkpoint1 = torch.load("best_loss.ckpt")
    checkpoint2 = torch.load(params["ckpt_dir"]+"/"+params["ckpt_name"]+"_minmaxval.ckpt")

    loader = SDFDataLoader(params["test_data_file"], 
                           params["kbond"], 
                           params["prop_list"], 
                           params["label_cols"],
                           False,params["theta_type"],
                           params["coarse_graining_params"])

    molecule_dataset_test = MoleculeDataset(loader.atom_feature_list, 
                                            loader.labels_list, 
                                            checkpoint2['maxval'], 
                                            checkpoint2['minval']
                                           )

    data_loader_test = data.DataLoader(molecule_dataset_test, batch_size=params["batch_size"], 
                                       shuffle=False, collate_fn=gcn_collate_fn)

    num_data = len(molecule_dataset_test)
    num_ckpt = len(ckpt_file_list)
#
#   LOOP for ckpt files
#
    arrays = np.zeros((num_data,num_ckpt+2))

    for ifile, ckpt_file in enumerate(ckpt_file_list): 
        checkpoint1 = torch.load(ckpt_file)

        model = GraphConvModel(device_ext=args.accelerator,
                               kbond=params["kbond"],
                               prop_list=params["prop_list"],
                               task=params["task"],
                               conv_layer_sizes=params["conv_layer_sizes"],
                               mlp_layer_sizes=params["mlp_layer_sizes"],
                               lr=params["lr"],
                               drop_ratio=params["drop_ratio"]
                              )
        model.load_state_dict(checkpoint1['state_dict'])
#
#       prediction
#
        model.eval()

        imol = -1
        with torch.no_grad():
            data1d_list = []
            for atom_features, atom_list, labels_list in data_loader_test: 
                atom_features = torch.tensor(np.array(atom_features), dtype = torch.float)
                labels        = torch.tensor(np.array(labels_list), dtype = torch.float)

                y, mlp_top = model(atom_list,atom_features)

                loss = F.mse_loss(y, labels)
                var_y = torch.var(labels, unbiased=False)
                r2 = 1.0 - F.mse_loss(y, labels, reduction="mean") / var_y

                data1d_list.extend(mlp_top.to("cpu").detach().numpy().copy())
                y = y.to("cpu").detach().numpy().copy()
                labels = labels.to("cpu").detach().numpy().copy()

                for i in range(len(y)):
                    imol = imol + 1
                    arrays[imol][ifile+1] = y[i][0]

                    if ifile == 0:
                        arrays[imol][0] = labels[i][0]
#
#  ensemble average
#
    for imol in range(num_data):
        tmp = 0.0
        for ifile in range(num_ckpt):
            tmp = tmp + arrays[imol][ifile+1]

        arrays[imol][num_ckpt+1] = tmp/float(num_ckpt)
#
#  save data
#
    colum_list = ['label']
    for i in range(num_ckpt):
        colum_list.append('ckpt'+str(i))
    colum_list.append('average')

    df = pd.DataFrame(data=arrays,columns=colum_list)
 
    df.to_csv('pred_results.csv')

if __name__ == "__main__":
    main()
