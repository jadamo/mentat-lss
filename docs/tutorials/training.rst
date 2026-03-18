.. _training:

Training on GPU(s)
===================

Training networks on GPUs has exploded in popularity in recent years. As such,
we have built mentat-lss with full gpu compatability on both linux and macos (arm64) architectures.
The hardest part for you is thus making sure your enviornment is correctly setup to utilize your hardware.

Enviornment troubleshooting
----------------------------

As a first step, you can (and should!) check after install that PyTorch knows about any gpus available. To do so,
run the following commands in a python terminal,

.. code-block:: python

    import torch
    # If on linux, run this command
    torch.cude.is_available() #<- should return True if correctly built for gpu
    # If on macos, run this command
    torch.backends.mps.is_available() #<- should return True if correctly built for gpu

If the above returns false **and you have a gpu available**, you probably installed the cpu build of PyTorch. That can happen
if you don't have ``cuda`` installed, which you can double-check by running ``nvidia-smi`` in the terminal. If you do have 
``cuda`` properly installed, follow the `pytorch local installation instructions`_.

.. _`pytorch local installation instructions`: https://pytorch.org/get-started/locally/

Once you've verified pytorch can see your gpu, you can simply run our example training script with
``python ./scripts/train_emulator.py``!

Multiple GPUs
-------------

The above example script will also attempt to train the emulator on multiple GPUs at once, potentially
saving a significant amount of time. It does this by assigning each sub-network in the emulator to one GPU only,
and periodically syncing up the results from all gpus together. Therefore, if you are running the above script then there
"should" be no extra work required for utilizing more than one gpu.

Example slurm script
--------------------

You will most likely want to train your emulator on an HPC system. To facilitate doing so,
we've proved an example slurm script below, based off running on University of Arizona systems.

.. code-block:: console

    #!/bin/bash
    #BATCH --job-name=<name of job>
    #SBATCH --mail-type=ALL
    #SBATCH --mail-user=<email> 
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=5
    #SBATCH --mem-per-cpu=8gb
    #SBATCH --gres=gpu:2 <- number of gpus PER NODE>
    #SBATCH --time=48:00:00   
    #SBATCH --partition=<queue name>
    #SBATCH --account=<account name>
    #SBATCH --output=<name_of_output_file>.out

    # load any system-specific modules (ex: anaconda)
    module load <modules>
    source /path/to/home/.bashrc

    # replace with your specific enviornment
    conda activate <anaconda_enviornment>

    cd /path/to/repo/scripts

    config_file="/path/to/net/config/file.yaml"
    echo "starting job"

    python train_emulator.py $config_file

Note the line ``#SBATCH --gres=gpu:2`` will request 2 GPUs on the same node.
Using gpus across different notes will be added in a future release, so do so at your own risk.