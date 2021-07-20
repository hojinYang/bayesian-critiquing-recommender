#!/usr/bin/env bash
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner AE.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner AE_sn.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_anneal.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_anneal_sn.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_vanilla.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_vanilla_sn.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_beta.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_beta_sn.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_optimal_nb.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_optimal.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_optimal_nb_sn.sh
#sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_optimal_sn.sh

#sbatch --nodes=1 --time=16:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_beta_multilayer.sh
#sbatch --nodes=1 --time=16:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAE_beta_multilayer_sn.sh
sbatch --nodes=1 --time=16:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_multilayer_optimal.sh
sbatch --nodes=1 --time=16:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_multilayer_optimal_sn.sh
sbatch --nodes=1 --time=16:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_multilayer_learn.sh
sbatch --nodes=1 --time=16:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_multilayer_learn_sn.sh
# sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_learn_global_nb.sh
# sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_learn_nb.sh
# sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_optimal_fixed_nb.sh
# sbatch --nodes=1 --time=12:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner VAEsigma_optimal_fixed.sh
