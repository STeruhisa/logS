import argparse
import runpy
import torch
from torch.utils import data
import pytorch_lightning as pl
import ax
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from simpleGCN.data.dataloader import SDFDataLoader
from simpleGCN.data.dataset import MoleculeDataset, gcn_collate_fn
from simpleGCN.models.graph_conv_model import GraphConvModel


def evaluation_function(parameters):

    # reading hyper parameters
    batch_size = parameters["batch_size"]
    conv_layer_width = parameters["conv_layer_width"]
    conv_layer_size = parameters["conv_layer_size"]
    mlp_layer_size = parameters["mlp_layer_size"]
    lr = parameters["lr"]
    drop_ratio = parameters["drop_ratio"]

###############################
#   set data_loader
###############################
    data_loader_train = data.DataLoader(params["molecule_dataset_train"], 
                                        batch_size=batch_size,
                                        drop_last=params["drop_last"],
                                        shuffle=False, collate_fn=gcn_collate_fn)

    data_loader_val = data.DataLoader(params["molecule_dataset_val"], 
                                      batch_size=batch_size,
                                      drop_last=params["drop_last"],
                                      shuffle=False, collate_fn=gcn_collate_fn)

    print("batch_size={0}".format(batch_size))
    print("conv_layer_width={0}".format(conv_layer_width))
    print("conv_layer_size={0}".format(conv_layer_size))
    print("mlp_layer_size={0}".format(mlp_layer_size))
    print("lr={0}".format(lr))
    print("drop_ratio={0}".format(drop_ratio))

    conv_layer_sizes = []
    for i in range(conv_layer_width):
        conv_layer_sizes.append(conv_layer_size)
##################################
#   MODEL
##################################
    model = GraphConvModel(device_ext=params["device"],
                           kbond=params["kbond"],
                           prop_list=params["prop_list"],
                           task=params["task"],
                           conv_layer_sizes=conv_layer_sizes,
                           mlp_layer_sizes=[mlp_layer_size, 1],
                           lr=lr,
                           drop_ratio=drop_ratio
                          )
##################################
#   initialize checkpoints
##################################
    key = "val_{0}".format(params["metrics"])
    if params["minimize"] :
        early_stop = EarlyStopping(monitor=key, patience=50, mode="min")
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=key, mode="min")
    else :
        early_stop = EarlyStopping(monitor=key, patience=50, mode="max")
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=key, mode="max")

##################################
#   fit the model
##################################
    trainer = pl.Trainer(max_epochs=params["num_epochs"],
                         gpus=params["gpu"],
                         callbacks=[early_stop, checkpoint_callback],
                         enable_progress_bar=False
                        )
    trainer.fit(model, data_loader_train, data_loader_val)

    best_score = checkpoint_callback.best_model_score.to('cpu').detach().numpy().copy()
    print("target_val={0}".format(best_score))
    print("last_val={0}".format(trainer.callback_metrics[key].item()))

    return float(best_score)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-hyper", action='store_true')
    args = parser.parse_args()

    global params

    params = runpy.run_path(args.config).get('model_params', None)
    pl.seed_everything(params["random_seed"])

    loader = SDFDataLoader(params["train_data_file"], 
                           params["kbond"], 
                           params["prop_list"], 
                           params["label_cols"]
                          )

    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(loader.atom_feature_list, 
                                                       loader.labels_list, 
                                                       shuffle=True, 
                                                       train_size=0.8, 
                                                       random_state=params["random_seed"]
                                                      )

    molecule_dataset_train = MoleculeDataset(X_train, y_train, loader.maxval, loader.minval)
    molecule_dataset_val   = MoleculeDataset(X_val, y_val, loader.maxval, loader.minval)

    if len(molecule_dataset_train) % params["batch_size"] == 1 : 
        drop_last = True
    else :
        drop_last = False

    device = torch.device('cpu')
    gpu = 0

#######################################################################
#
#   HYPER PARAMETER FITTING
#
#######################################################################
    if args.hyper:
        params["molecule_dataset_train"] = molecule_dataset_train
        params["molecule_dataset_val"] = molecule_dataset_val
        params["device"] = device
        params["gpu"] = gpu
        params["drop_last"] = drop_last

        best_parameters, best_values, experiment, model = ax.optimize(
                              params["hyper"]["parameters"],
                              evaluation_function,
                              random_seed=params["random_seed"],
                              minimize=params["minimize"],
                              total_trials=params["hyper"]["trials"]
                              )

        print(best_parameters)
#######################################################################
#
#   STANDARD TRAINING
#
#######################################################################
    else:
        data_loader_train = data.DataLoader(molecule_dataset_train, batch_size=params["batch_size"],
                                            drop_last=drop_last, shuffle=False, 
                                            collate_fn=gcn_collate_fn)

        data_loader_val = data.DataLoader(molecule_dataset_val, batch_size=params["batch_size"],
                                          drop_last=drop_last, shuffle=False, 
                                          collate_fn=gcn_collate_fn)

        model = GraphConvModel(device_ext=device,
                               kbond=params["kbond"],
                               prop_list=params["prop_list"],
                               task=params["task"],
                               conv_layer_sizes=params["conv_layer_sizes"],
                               mlp_layer_sizes=params["mlp_layer_sizes"],
                               lr=params["lr"],
                               drop_ratio=params["drop_ratio"]
                              )
###############################
#       call back
###############################
        loss_checkpoint = ModelCheckpoint(
                          dirpath='./',
                          filename=f"best_loss",
                          monitor="val_loss",
                          save_last=True,
                          save_top_k=1,
                          save_weights_only=True,
                          mode="min",
                                          )
###############################
#       training
###############################
        trainer = pl.Trainer(max_epochs=params["num_epochs"],
                             gpus=gpu,
                             callbacks=[loss_checkpoint]
                            )
        trainer.fit(model, data_loader_train, data_loader_val)
####################################
#       save checkpoint
####################################
        ckptfile = 'minmaxval.ckpt'
        torch.save({'maxval': loader.maxval,
                    'minval': loader.minval,
                    'average': loader.avg
                    }, ckptfile)


if __name__ == "__main__":
    main()
