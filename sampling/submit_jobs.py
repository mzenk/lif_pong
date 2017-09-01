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
    sampling_method = 'lif'
    pot_str = 'pong'
    win_size = 48
    chunk_size = 5
    n_total = 200
    starts = range(0, n_total, chunk_size)
    print("Send sampling jobs to slurm")
    jobs = []

    memory_opt = "--mem=8G"
    for start in starts:
        args = (sampling_method, pot_str, win_size, start, chunk_size)
        jobs.append(
            submit_job("python {}_sampling.py {} {} {} {}".format(*args),
                       memory_opt)
            )

    args = (pot_str, win_size)
    dependency = "--kill-on-invalid-dep=yes" \
        " --dependency=afterok:" + ":".join(jobs)
    submit_job("python save_prediction_data.py {} {}".format(*args),
               dependency)
    # submit_job("python clean_up_sample_data.py {} {}".format(*args), dependency)
