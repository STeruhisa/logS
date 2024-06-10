model_params = {
#
#   model parameter
#
    'task': 'regression', # 'regression' or 'classification'
    'random_seed': 42,
    'num_epochs': 200,
    'train_data_file': 'input/HF_train_SOLV.sdf', # input file path
    'test_data_file': 'input/HF_test_SOLV.sdf',
    'ckpt_dir': './dirckpt',
    'ckpt_name': 'IAP',  # ex. 'ckpt_name'_best_loss_0.ckpt
    'theta_type': 2, # activation for atomic descripters (0: none, 1: ReLu, 2: Softplus)
#######################################################################
#   target
#######################################################################
    'label_cols': ['SOLV_exp'], # target parameter label
#######################################################################
#   atomic features
#######################################################################
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
###########################################################################
#   hyper parameters
###########################################################################
    'kbond': 1,
    'batch_size': 137,
    'conv_layer_sizes': [111, 111, 111],  # convolution layer sizes
    'mlp_layer_sizes': [191, 1], # multi layer perceptron sizes
    'lr': 0.05158258039555578, #learning late
    'drop_ratio': 0.21427010541802763,
##########################################################################
#   parameter for Hyperparameter fitting
##########################################################################
    'metrics': 'r2', # the metrics for 'check_point' , 'early_stopping', 'hyper'
    'minimize': False, # True if you want to minimize the 'metrics'
    'hyper':
        {
         'trials': 100,
         'parameters':
             [
              {'name': 'kbond', 'type': 'int', 'bounds': [1, 7], 'value_type': 'int'},
              {'name': 'batch_size', 'type': 'range', 'bounds': [50, 300], 'value_type': 'int'},
              {'name': 'conv_layer_width', 'type': 'range', 'bounds': [1, 4], 'value_type': 'int'},
              {'name': 'conv_layer_size', 'type': 'range', 'bounds': [5, 200], 'value_type': 'int'},
              {'name': 'mlp_layer_size', 'type': 'range', 'bounds': [5, 200], 'value_type': 'int'},
              {'name': 'lr', 'type': 'range', 'bounds': [0.001, 0.1], 'value_type': 'float'},
              {'name': 'drop_ratio', 'type': 'range', 'bounds': [0.0, 1.0], 'value_type': 'float'},
             ]
        }
}


