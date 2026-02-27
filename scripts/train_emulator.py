from mentat_lss.emulator import ps_emulator, compile_multiple_device_training_results
import time, sys
import torch
import torch.multiprocessing as mp
import itertools
import logging

import mentat_lss.training_loops as training_loops

def main():

    if len(sys.argv) < 2:
        print("USAGE: python train_emulator.py <config_file>")
        return 0

    logging.basicConfig(level = logging.INFO)
    logger = logging.getLogger("train_emulator")

    t1 = time.time()
    emulator = ps_emulator(sys.argv[1], "train")

    # train on a single cpu / gpu
    if emulator.num_gpus < 2:
        logger.info(f"Training network on device: {emulator.device.type}")
        training_loops.train_on_single_device(emulator)

    # split the sub-networks to train on multiple gpus
    else:
        logger.info("Splitting up training on {:d} GPUs...".format(emulator.num_gpus))
        # spawn() usually behaves better than fork() on HPC
        mp.set_start_method("spawn", force=True)
        if emulator.model_type == "combined_tracer_transformer":
            net_idx = torch.Tensor(list(range(emulator.num_zbins))).to(int)
        else:
            net_idx = torch.Tensor(list(itertools.product(range(emulator.num_spectra), range(emulator.num_zbins)))).to(int)
        split_indices = net_idx.chunk(emulator.num_gpus)
        
        # spawn() usually behaves better than fork() on HPC
        mp.spawn(
            training_loops.train_on_multiple_devices,
            args=(split_indices, sys.argv[1]),
            nprocs=emulator.num_gpus,
            join=True
        )
        full_emulator = compile_multiple_device_training_results(emulator.input_dir + emulator.save_dir, sys.argv[1], emulator.num_gpus)
        full_emulator._save_model()

    t2 = time.time()
    logger.info("Training Done in {:0.0f} hours {:0.0f} minutes {:0.2f} seconds\n".format(
        (t2-t1)//3600, ((t2-t1) % 3600) // 60, (t2-t1)%60))

if __name__ == "__main__":
    main()
