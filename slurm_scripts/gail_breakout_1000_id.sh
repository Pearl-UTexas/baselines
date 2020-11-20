#!/bin/bash

#SBATCH --job-name GAIL_breakout_id
#SBATCH --output=slurm_logs/slurmjob_%j.out
#SBATCH --error=slurm_logs/slurmjob_%j.err
#SBATCH --mail-user=asaran@cs.utexas.edu
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --partition titans
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time 84:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
mpirun -np 4 python -m baselines.gail.run_atari --env_id BreakoutNoFrameskip-v4 --expert_path ./data/critical_states/new_breakout_1000 --log_dir ./log/breakout_1000_cs/id --cs id