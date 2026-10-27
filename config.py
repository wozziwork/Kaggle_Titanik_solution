from pathlib import Path
from omegaconf import OmegaConf


config = {
    'general' : {
        'experiment_name' : 'all_progon',
        'seed' : 67,
        'train' : True,
        'inference' : False,
        'save_models' : False,
        'save_logs' : False,
        'boosting_verbose' : 0,
        'device' : 'cpu',
        'show_graphs' : True
    },
    'paths' : {
        'TRAIN_PATH' : Path(__file__).resolve().parent / "data" / "train.csv",
        'TEST_PATH' : Path(__file__).resolve().parent / "data" / "test.csv",
        'MODELS_DIR' : Path(__file__).resolve().parent / "saved_models"
    },
    'dataset' : {
        'dropped_columns' : ['PassengerId', 'Ticket', 'Name', 'Cabin'],
        'numeric_columns' : ['Age', 'Fare', 'SibSp', 'Parch'],
        'categorical_columns' : ['Sex', 'Embarked', 'Pclass', 'Title'],
    },
    'preprocessing' : {
        'numeric' : {
            'strategy' : 'median',
        },
        'categorical' : {
            'strategy' : 'constant',
            'fill_value' : 'Unknown',
        }
    },
    'validation' : {
        'n_splits' : 5,
        'shuffle' : True,
        'scorers' : ['accuracy','f1']
    },
    'models' : {
        'logistic_regression' : {
            'C' : 0.2,
            'solver' : 'saga',
            'max_iter' : 1000,
            'l1_ratio' : 0.
        },
        'knn' : {
            'n_neighbors' : 9,
            'metric' : 'manhattan',
            'weights' : 'uniform',
            'algorithm' : 'brute'
        },
        'decision_tree' : {
            'max_depth' : 10,
            'random_state' : '${general.seed}',
            'min_samples_split' : 20,
            'min_samples_leaf' : 5,
            'criterion' : 'entropy',
            'splitter' : 'best',
            'max_features' : None,
        },
        'random_forest' : {
            'max_depth' : 9,
            'max_features' : None,
            'min_samples_leaf' : 2,
            'min_samples_split' : 5,
            'n_estimators' : 80,
            'n_jobs' : 1,
            'random_state' : '${general.seed}',
        },
        'catboost' : {
            'random_seed' : '${general.seed}',
            'cat_features' : ['Sex', 'Embarked', 'Pclass', 'Title'],
            'verbose' : '${general.boosting_verbose}',
            'thread_count': 1,
            'early_stopping_rounds' : 40,
            'use_best_model' : True,
            'iterations' : 500,
            'max_depth' : 6,
            'learning_rate' : 0.05,
            'l2_leaf_reg' : 3,
            'random_strength' : 0,
            'subsample' : 0.8,
            'bootstrap_type': 'MVS',
        },
        'xgboost' : {
            'random_state' : '${general.seed}',
            'n_jobs' : 1,
            'enable_categorical' : True,
            'early_stopping_rounds' : 80,
            'tree_method' : 'hist',
            'eval_metric' : 'logloss',
            'n_estimators' : 1000,
            'learning_rate' : 0.05,
            'max_depth' : 6,
            'reg_lambda' : 1.0,
            'reg_alpha' : 0.0,
            'colsample_bytree' : 1.,
            'subsample' : 1.,
            'gamma' : 0.,
            'min_child_weight' : 1,
        },
        'lightgbm' : {
            'random_state': '${general.seed}',
            'verbosity': -1,
            'n_jobs': 1,
            'early_stopping_round' : 80,
            'learning_rate': 0.05,
            'num_leaves' : 17,
            'n_estimators': 1000,
            'reg_lambda' : 1.0,
            'min_child_samples' : 22,
            'subsample' : 1.,
        },
        'mlp' : {
            'hidden_dims' : [32,32,32],
            'batch_norm' : True,
            'dropout' : 0.,
            'activation_func' : 'leacky_relu'
        },
        'mlp_emb' : {
            'hidden_dims' : [128, 128],
            'embedding_dims' : [12, 12, 16, 4],
            'batch_norm': True,
            'dropout': 0.3125,
            'activation_func': 'relu'
        }
    },
    'dl_train' : {
        'mlp' : {
            'num_workers' : 0,
            'epochs' : 400,
            'batch_size' : 64,
            'lr' : 0.00217,
            'optimizer_name': 'adam',
            'weight_decay': 0.01,
            'early_stopping_patience' : 10,
        },
        'mlp_emb' : {
            'num_workers': 0,
            'epochs': 400,
            'batch_size': 24,
            'lr': 0.002,
            'optimizer_name': 'adamw',
            'weight_decay': 0.26,
            'early_stopping_patience': 10,
        }
    },
    'search' : {
        'summarize_results': False,
        'refit': 'accuracy',
        'n_jobs' : -1,
        'verbose' : 0,
    },
    'param_grids' : {
        'logistic_regression' : {
            'model__C' : [0.2, 0.3],
            'model__l1_ratio' : [0., 0.5, 1.]
        },
        'knn' : {
            'model__n_neighbors' : [1,3,5,7],
            'model__metric' : ['manhattan'],
            'model__weights' : ['uniform', 'distance'],
        },
        'decision_tree' : {
            'model__max_depth' : [3,4,5,10,20],
            'model__min_samples_split' : [19,20,21],
            'model__min_samples_leaf' : [5,6,7,8,10,20],
        },
        'random_forest' : {
            'model__n_estimators' : [65,70,75,80],
            'model__max_depth' : [9,10,11],
            'model__min_samples_split' : [3,4,5],
        },
        'catboost' : {
            'model__max_depth' : [4,5,6,7],
            'model__learning_rate' : [0.03, 0.05, 0.075],
            'model__l2_leaf_reg' : [2,3,4],

        },
        'xgboost' : {
            'model__min_child_weight': [1, 3, 5, 10],
            'model__subsample' : [0.5, 0.75, 0.8, 1],
            'model__colsample_bytree': [0.6, 0.7, 0.8, 1],
            'model__gamma' : [0, 0.1, 0.5, 1.]
        },
        'lightgbm': {
            'model__num_leaves': [16],
            'model__learning_rate': [0.05],
            'model__min_child_samples': [22],
            'model__reg_lambda': [1],
            'model__subsample': [1.0],
            'model__subsample_freq': [1],
            'model__colsample_bytree': [1.0],
        },
        'mlp': {

            'hidden_dims': [[32,32,32]],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            'lr': [0.002],
            'activation_func': ['leacky_relu'],
            'batch_norm': [True],
            'batch_size': [24],
            'optimizer_name': ['adam'],
            'weight_decay': [0.0, 0.1, 0.01],
        },
        'mlp_emb': {
            'dropout': [0.1, 0.2, 0.3],
            'lr': [0.0015, 0.002, 0.0025],
            'hidden_dims': [[128,128]],
            'batch_norm': [True],
            'batch_size': [24],
            'activation_func': ['leacky_relu'],
            'optimizer_name': ['adam'],
            'weight_decay': [0.01],
        },
    },
    'to_train' : {
        'logistic_regression' : False,
        'knn' : False,
        'decision_tree' : False,
        'random_forest' : False,
        'catboost' : False,
        'xgboost' : False,
        'lightgbm' : False,
        'mlp' : False,
        'mlp_emb' : True,
    },
    'to_grid_search' : {
        'logistic_regression' : False,
        'knn' : False,
        'decision_tree' : False,
        'random_forest' : False,
        'catboost' : False,
        'xgboost' : False,
        'lightgbm' : False,
        'mlp' : False,
        'mlp_emb' : False,
    },
    'inference' : {
          'logistic_regression' : 'baseline_logistic_regression',
    },
    'to_inference' : {
        'logistic_regression' : False,
        'knn' : False,
        'decision_tree' : False,
        'random_forest' : False,
        'catboost' : False,
        'xgboost' : False,
        'mlp' : False,
        'mlp_emb' : False,
    }

}


config = OmegaConf.create(config)

