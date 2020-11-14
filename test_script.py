import os, sys

bm4d_bin = './RIBM4D/ribm4d-gpu'

def launch_job(executable, noisy_fl, gt_fl, out_fl='./tmp', sim_th=2500.0, hard_th=2.7, window_size=5, step_size=3, special_info='RIBM4D'):
    job = " ".join([str(x) for x in [executable, noisy_fl, gt_fl, out_fl, sim_th, hard_th, window_size, step_size]])
    sbatch_job = """ python3 sbatch_run.py --exclude="gpu-compute[1-3]" -x="gpu-s" --specialinfo="LOG_{}" -r"{}" """.format(special_info, job)
    print(sbatch_job)
    os.system(sbatch_job)

launch_job(bm4d_bin, "./data/3j7h_cropped_center_avi/3j7h_center_noisy_0.20.avi", "./data/3j7h_cropped_center_avi/3j7h_center_original.avi", special_info="3j7h_center_noisy_0.20")

# TODO: Write a for loop?