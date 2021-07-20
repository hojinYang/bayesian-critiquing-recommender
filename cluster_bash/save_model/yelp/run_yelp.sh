#!/usr/bin/env bash
sbatch --nodes=1 --time=1:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_beta_multilayer.sh
