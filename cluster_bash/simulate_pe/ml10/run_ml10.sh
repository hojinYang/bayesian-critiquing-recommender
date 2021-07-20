#!/usr/bin/env bash
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg0_noise0.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise0.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise3.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise5.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise8.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg1_noise10.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg10_noise0.sh
#sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_abs_diff_neg100_noise0.sh

#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg0_noise0.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise0.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise3.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise5.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise8.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg1_noise10.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg10_noise0.sh
#sbatch --nodes=1 --time=5:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner sim_pop_neg100_noise0.sh

sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold00.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold03.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold05.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold10.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold13.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold15.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold20.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold23.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold25.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold30.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold33.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold35.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold40.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold43.sh
sbatch --nodes=1 --time=3:00:00 --mem=32G --cpus-per-task=4 --gres=gpu:1 --account=rrg-ssanner fold45.sh