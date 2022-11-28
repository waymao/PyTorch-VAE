from grid_search_config import CMD, NUM_PARALLEL_JOBS, PARAMS_TO_SEARCH
from multiprocessing.pool import ThreadPool
import subprocess

params_list = []

def worker(params):
    ident = ", ".join(params[1::2])
    print(f'Working on {ident}')
    process = subprocess.Popen(
        [*CMD, *params], 
        stdout=subprocess.PIPE, 
        universal_newlines=True,
        bufsize=8192
    )
    for line in process.stdout:
        if "on test" in line and "per-pixel VI-loss" in line:
            print(ident + " | " + line.strip())
    print(f'Finished {ident}')


for config in PARAMS_TO_SEARCH['config']:
    params_list.append([
        "--config", str(config),
    ])

tpool = ThreadPool(NUM_PARALLEL_JOBS)
for params in params_list:
    tpool.apply_async(worker, (params,))

tpool.close()
tpool.join()
