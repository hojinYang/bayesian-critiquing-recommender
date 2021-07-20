#!/usr/bin/env bash
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg0_noise0.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise0.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise3.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise5.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise8.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise10.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg10_noise0.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg100_noise0.sh
sbatch --nodes=1 --time=1:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold0.sh
sbatch --nodes=1 --time=1:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold1.sh
sbatch --nodes=1 --time=1:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold2.sh
sbatch --nodes=1 --time=1:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold3.sh
sbatch --nodes=1 --time=1:30:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold4.sh

#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg0_noise0.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise0.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise3.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise5.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise8.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise10.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg10_noise0.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg100_noise0.sh