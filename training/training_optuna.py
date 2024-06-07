import argparse
import runpy
import torch
from torch.utils import data
import pytorch_lightning as pl
import optuna
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from moduleNN.data.dataloader import SDFDataLoader
from moduleNN.data.dataset import MoleculeDataset, gcn_collate_fn, KFoldDataModule, KFoldLoop
from moduleNN.models.graph_conv_model import GraphConvModel


def objective(trial):

##############################
#   reading hyper parameters
##############################
    hp_parameters_list = params["hyper"]["parameters"]
    num_hp = len(hp_parameters_list)

    hyperparameters = {}
    for i, hp_parameters in enumerate(hp_parameters_list):
        key = hp_parameters["name"]
        value_type = hp_parameters["value_type"]
        par_min = hp_parameters["bounds"][0]
        par_max = hp_parameters["bounds"][1]

        if value_type == 'int':
            hyperparameters[key] = trial.suggest_int(key, par_min, par_max)
        else :
            hyperparameters[key] = trial.suggest_float(key, par_min, par_max)

        print(key,"= {0}".format(hyperparameters[key]))
###############################
#   set data loader
###############################
    loader = SDFDataLoader(params["train_data_file"],
                           hyperparameters["kbond"],
                           params["prop_list"],
                           params["label_cols"],
                           False,params["theta_type"],
                           params["coarse_graining_params"]
                          )
###############################
#   set data module
###############################
    dm = KFoldDataModule(features = loader.atom_feature_list,
                         labels = loader.labels_list,
                         maxval = loader.maxval,
                         minval = loader.minval,
                         batch_size = hyperparameters["batch_size"],
                         collate_fn = gcn_collate_fn)

    conv_layer_sizes = []
    for i in range(hyperparameters['conv_layer_width']):
        conv_layer_sizes.append(hyperparameters['conv_layer_size'])
##################################
#   MODEL
##################################
    model = GraphConvModel(device_ext=params["accelerator"],
                           kbond=hyperparameters["kbond"],
                           prop_list=params["prop_list"],
                           task=params["task"],
                           conv_layer_sizes=conv_layer_sizes,
                           mlp_layer_sizes=[hyperparameters['mlp_layer_size'], 1],
                           lr=hyperparameters['lr'],
                           drop_ratio=hyperparameters['drop_ratio']
                          )
##################################
#   initialize checkpoints
##################################
    key = "val_{0}".format(params["metrics"])
    if params["minimize"] :
        early_stop = EarlyStopping(monitor=key, patience=50, mode="min")
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=key, mode="min",
                                              dirpath=params["ckpt_scr"])
    else :
        early_stop = EarlyStopping(monitor=key, patience=50, mode="max")
        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=key, mode="max",
                                              dirpath=params["ckpt_scr"])

##################################
#   fit the model
##################################
    trainer = pl.Trainer(max_epochs=params["num_epochs"],
                         accelerator=params["accelerator"],
                         devices=params["devices"],
                         num_sanity_val_steps=0,
                         callbacks=[early_stop, checkpoint_callback],
                         enable_progress_bar=False,
                         logger=False
                        )
#   trainer.logger.log_hyperparams(hyperparameters)

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = KFoldLoop(5, export_path="./")
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, dm)

    best_score = trainer.fit_loop.best_score
    print("target_val={0}".format(best_score))
#   print("last_val={0}".format(trainer.callback_metrics[key].item()))

    return float(best_score)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str, required=True)
    parser.add_argument("-ckpt_scr", type=str, required=True)
    parser.add_argument("-hyper", action='store_true')
    parser.add_argument("-accelerator", default=None)
    parser.add_argument("-devices", default=None)
    args = parser.parse_args()

    global params

    params = runpy.run_path(args.config).get('model_params', None)
    pl.seed_everything(params["random_seed"])

#   loader = SDFDataLoader(params["train_data_file"], 
#                          params["kbond"], 
#                          params["prop_list"], 
#                          params["label_cols"],
#                          False,params["theta_type"],
#                          params["coarse_graining_params"]
#                         )

#######################################################################
#
#   HYPER PARAMETER FITTING
#
#######################################################################
    if args.hyper:
        params["devices"] = args.devices
        params["accelerator"] = args.accelerator
        params["ckpt_scr"] = args.ckpt_scr

        if params["minimize"] :
            direction = "minimize"
        else : 
            direction = "maximize"

        study = optuna.create_study(
                                    study_name="ML-study",
                                    sampler=optuna.samplers.TPESampler(),
                                    pruner=optuna.pruners.MedianPruner(),
                                    direction=direction
                                   )
        study.optimize(objective, n_trials=params["hyper"]["trials"])
 
#######################################################################
#
#   STANDARD TRAINING
#
#######################################################################
    else:
#
#       set dataloader
#
        loader = SDFDataLoader(params["train_data_file"],
                               params["kbond"],
                               params["prop_list"],
                               params["label_cols"],
                               False,params["theta_type"],
                               params["coarse_graining_params"]
                              )
#
#       ser data module & model
#
        dm = KFoldDataModule(features = loader.atom_feature_list,
                             labels = loader.labels_list,
                             maxval = loader.maxval,
                             minval = loader.minval,
                             batch_size = params["batch_size"],
                             collate_fn = gcn_collate_fn)
  
        model = GraphConvModel(device_ext=args.accelerator,
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
        key = "val_{0}".format(params["metrics"])
        if params["minimize"] :
            mode_direction = "min"
        else:
            mode_direction = "max"

        loss_checkpoint = ModelCheckpoint(
                          dirpath='./',
                          filename=params["ckpt_dir"]+"/"+params["ckpt_name"]+"_best_loss",
                          monitor=key,
                          save_last=True,
                          save_top_k=1,
                          save_weights_only=True,
                          mode=mode_direction,
                                          )
###############################
#       training
###############################
        trainer = pl.Trainer(max_epochs=params["num_epochs"],
                             accelerator=args.accelerator, 
                             devices=args.devices,
                             num_sanity_val_steps=0, 
                             callbacks=[loss_checkpoint],
                             logger=False
                            )
        internal_fit_loop = trainer.fit_loop
        trainer.fit_loop = KFoldLoop(5, export_path="./")
        trainer.fit_loop.connect(internal_fit_loop)
        trainer.fit(model, dm)
####################################
#       save checkpoint
####################################
        ckptfile = params["ckpt_dir"]+"/"+params["ckpt_name"]+'_minmaxval.ckpt'
        torch.save({'maxval': loader.maxval,
                    'minval': loader.minval,
                    'average': loader.avg
                    }, ckptfile)


if __name__ == "__main__":
    main()
