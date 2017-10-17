#! /usr/bin/env python

from __future__ import print_function
import commands


# parser for options would be good
def submit_job(wrap_cmd, sbatch_options=''):
    cmd = "sbatch -p simulation {} --wrap=\"{}\"".format(sbatch_options,
                                                         wrap_cmd)
    print("Submitting job with command: %s " % cmd)
    status, job_msg = commands.getstatusoutput(cmd)
    if status == 0:
        print(job_msg)
    else:
        print("Error submitting job.")
    return job_msg.split(' ')[-1]

if __name__ == '__main__':
    pot_str = 'pong'
    win_size = 48
    chunk_size = 20
    n_total = 1000
    starts = range(0, n_total, chunk_size)
    print("Send sampling jobs to slurm")
    jobs = []
    script_name = 'lif_performance'
    save_modifier = ''

    memory_opt = "--mem=8G"
    for start in starts:
        args = \
            (script_name, pot_str, start, chunk_size, win_size, save_modifier)
        wrap_cmd = 'python {}.py {} {} {} {} {}'.format(*args)
        jobs.append(submit_job(wrap_cmd, memory_opt))

    args = (pot_str, win_size, script_name, save_modifier)
    dependency = "--kill-on-invalid-dep=yes" \
        " --dependency=afterok:" + ":".join(jobs)
    submit_job('python lif_save_prediction_data.py {} {} {} {}'.format(
               *args), dependency)
