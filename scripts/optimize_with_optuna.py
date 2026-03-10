import os
import logging
import optuna
import shutil
import yaml
import argparse
import torch 

import mentat_lss.training_loops as training_loops
from mentat_lss.utils import load_config_file
from mentat_lss.emulator import ps_emulator

def create_cache(node_id=0):
    cache_dir = f"cache_node{node_id}"
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

def clean_cache(node_id=0):
    cache_dir = f"cache_node{node_id}"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

def define_model(trial, node_id, default_config_file, device=None):

    trial_config_file = default_config_file.copy()

    num_mlp_blocks = trial.suggest_int("num_mlp_blocks", 1, 6)
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 0, 1)
    num_block_layers = trial.suggest_int("num_block_layers", 2, 6)
    batch_size = trial.suggest_int("batch_size", 100, 1000)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2)

    trial_config_file["galaxy_ps_emulator"]["num_mlp_blocks"] = num_mlp_blocks
    trial_config_file["galaxy_ps_emulator"]["num_block_layers"] = num_block_layers
    trial_config_file["galaxy_ps_emulator"]["num_transformer_blocks"] = num_transformer_blocks
    trial_config_file["batch_size"] = batch_size
    trial_config_file["galaxy_ps_learning_rate"] = learning_rate
    # to save time, only train with 10% of the full data
    trial_config_file["training_set_fraction"] = 0.1

    cache_dir = f"cache_node{node_id}"
    file_path = os.path.join(cache_dir, f'config_{trial.number}.yaml')
    with open(file_path, 'w') as outfile:
        yaml.dump(dict(trial_config_file), outfile, sort_keys=False, default_flow_style=False)

    return ps_emulator(file_path, "train", device=device)

def save_best_params(save_loc, default_config_file, best_params):
    best_config_file = default_config_file.copy()

    best_config_file["galaxy_ps_emulator"]["num_mlp_blocks"] = best_params["num_mlp_blocks"]
    best_config_file["galaxy_ps_emulator"]["num_block_layers"] = best_params["num_block_layers"]
    best_config_file["galaxy_ps_emulator"]["num_transformer_blocks"] = best_params["num_transformer_blocks"]
    best_config_file["batch_size"] = best_params["batch_size"]
    best_config_file["galaxy_ps_learning_rate"] = best_params["learning_rate"]

    dir_name = os.path.dirname(save_loc)
    base_name = os.path.splitext(os.path.basename(save_loc))[0]
    file_path = os.path.join(dir_name, f'{base_name}_optimized.yaml')
    print(f"Saving optimized hyperparameters to {file_path}...")
    with open(file_path, 'w') as outfile:
        yaml.dump(dict(best_config_file), outfile, sort_keys=False, default_flow_style=False)

def objective(trial, node_id, default_config_file, device=None):
    emulator = define_model(trial, node_id, default_config_file, device)

    training_loops.train_on_single_device(emulator)

    # We're using the average of the best losses for each sub-network
    best_losses = [min(losses) for losses in emulator.valid_loss if len(losses) > 0]
    result = sum(best_losses) / len(best_losses)

    # clean up trial save directory to avoid filling disk
    trial_save_dir = os.path.join(emulator.input_dir, emulator.save_dir)
    if os.path.exists(trial_save_dir):
        shutil.rmtree(trial_save_dir)

    return result

def run_worker(node_id, gpu_id, default_config_file, n_trials, max_time, db_path):
    
    device = torch.device(f"cuda:{gpu_id}")
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{db_path}",
        engine_kwargs={"connect_args": {"timeout": 30}}  # wait up to 30s for lock
    )
    study = optuna.load_study(
        study_name="combined_tracer",
        storage=storage
    )
    study.optimize(
        lambda trial: objective(trial, node_id, default_config_file, device),
        n_trials=n_trials,
        timeout=max_time
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="<Required> Path to config file."
    )
    parser.add_argument(
        "n_trials", type=int, default=None,
        help="<Required> number of trials to execute by Optuna."
    )
    parser.add_argument(
        "--db-path", type=str, default="optuna_results.db",
        help="Path to Optuna SQLite DB. Must be on a shared filesystem for multi-node."
    )
    command_line_args = parser.parse_args()

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger("")
    try:
        node_id = int(os.environ.get("SLURM_NODEID", 0))
    except (ValueError, TypeError):
        node_id = 0
        logger.warning("Failed to parse SLURM_NODEID, defaulting to 0")
    num_gpus = torch.cuda.device_count() # <- GPUs on the given node
    db_path = command_line_args.db_path

    # Create study once (workers load it)
    optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{db_path}",
        study_name="combined_tracer",
        pruner=optuna.pruners.MedianPruner(),
        load_if_exists=True
    )

    logger.info(f"Node {node_id} creating cache directory...")
    create_cache(node_id)

    default_config_file = load_config_file(command_line_args.config_file)
    max_time = int(2.9 * 24 * 60 * 60) # <- max of 3 days

    if num_gpus < 2:
        # Single GPU / CPU fallback
        device = torch.device("cuda:0") if num_gpus == 1 else torch.device("cpu")
        logger.info(f"Running trials on device {device}")
        study = optuna.load_study(study_name="combined_tracer", storage=f"sqlite:///{db_path}")
        study.optimize(
            lambda trial: objective(trial, node_id, default_config_file, device),
            n_trials=command_line_args.n_trials,
            timeout=max_time
        )
    else:
        import torch.multiprocessing as mp
        logger.info(f"Running trials on {num_gpus} GPUs with multiprocessing...")
        gpu_id = range(num_gpus)
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            run_worker,
            args=(node_id, gpu_id, default_config_file, command_line_args.n_trials // num_gpus, max_time, db_path),
            nprocs=num_gpus,
            join=True
        )
    study = optuna.load_study(study_name="combined_tracer", storage=f"sqlite:///{db_path}")

    if node_id == 0:
        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        save_best_params(command_line_args.config_file, default_config_file, trial.params)
    
    clean_cache(node_id)

if __name__ == "__main__":
    main()
