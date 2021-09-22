import math
import pprint

def return_boilerplate(jobid, t, edges, pus, tasks):
    approxtime = math.ceil(32 * float(edges) / 4000000.) + 1
    boilerplate = "" \
        + "#!/bin/bash\n" \
        + "#SBATCH --job-name=ac-rw-{}\n".format(2 * jobid + t - 1) \
        + "#SBATCH --partition=general1\n" \
        + "#SBATCH --nodes=1\n" \
        + "#SBATCH --ntasks={}\n".format(tasks) \
        + "#SBATCH --cpus-per-task={}\n".format(pus) \
        + "#SBATCH --mem-per-cpu={}\n".format(int(188. / float(tasks) * 1000)) \
        + "#SBATCH --time={}:00:00\n".format(approxtime) \
        + "#SBATCH --no-requeue\n" \
        + "#SBATCH --mail-type=FAIL\n" \
        + "\n" \
        + "DIR=\"/scratch/memhierarchy/penschuck/networks/network-repository.com/\"\n" \
        + "OUTPUTDIR=\"/scratch/memhierarchy/tran/parrwautocorrelation/$SLURM_JOB_ID/\"\n" \
        + "mkdir -p $OUTPUTDIR\n" \
        + "export OMP_NUM_THREADS={}\n".format(pus) \
        + "cp /home/memhierarchy/tran/edge-switching/release/autocorrelation_realworld .\n"
    return boilerplate

types = ["bio", "bn", "ca", "cit", "eco", "econ", "email", "heter", "ia", "inf", 
         "labeled", "massive", "power", "protein", "proximity", "rec", 
         "retweet_graphs", "road", "sc", "soc", "socfb", "tech", "web"]

inputdir = "/scratch/memhierarchy/penschuck/networks/network-repository.com"
outputdir = "/scratch/memhierarchy/tran/parrwautocorrelation"
jobs = 15
runs = 15
minsnaps = 10000
maxsnaps = 10000
tasks = 40
pus = 1
L = []

with open("sorted.csv", "r") as reader:
    for line in reader.read().splitlines()[1::]:
        s = line.split(",")
        n = int(s[1])
        m = int(s[2])
        stype = s[0].split("/")[0]
        if not (stype in types):
            continue
        # do 300k for now
        if m > 10000:
            continue
        if m < 1000:
            continue
        L.append((s[0], n, m))

print("chose graphs #", len(L))
    
L.sort(key=lambda x: x[2])
total_edges = sum(map(lambda x: x[2], L))
edges_by_jobs = total_edges / jobs

print("edges per job", math.floor(edges_by_jobs))

J = [[] for i in range(jobs)]
current_sum = 0
index = 0
for t in L:
    if current_sum + t[2] > edges_by_jobs:
        index = index + 1
        current_sum = t[2]
    else:
        current_sum = current_sum + t[2]
    if index >= len(J):
        index = 0
    J[index].append(t)

def fn(jobgraph):
    return jobgraph[0]

def logfn(fullpathfn):
    return fullpathfn[0].split("/")[-1].split(".simp-undir-edges")[0]

def chunks(lst, sz):
    for i in range(0, len(lst), sz):
        yield lst[i:i + sz]

thins = "1 2 3 4 5 6 7 8 9 10 12 14 15 18 20 21 24 25 26 28 30"
for Jobgraphs, index in zip(J, range(len(J))):
    if len(Jobgraphs) == 0:
        continue   
    jsum = sum(map(lambda k:k[2], Jobgraphs))
    for typ in [1, 2]:
        with open("job-{}.sh".format(2 * index + typ - 1), "w") as writer:
            JobgraphsDups = [(val, dupid) for val in Jobgraphs for dupid in range(runs)]
            todo_tasks = len(JobgraphsDups)
            job_bundles = list(chunks(JobgraphsDups, tasks))
            max_bundle_size = max(map(lambda k: len(k), job_bundles))
            writer.write(return_boilerplate(index, typ, jsum, pus, max_bundle_size))
            for bundle, bundleid in zip(job_bundles, range(len(job_bundles))):
                writer.write("echo \"at bundle {} / {}\"\n".format(bundleid, len(job_bundles) - 1))
                for graph, dupid in bundle:
                    writer.write("./autocorrelation_realworld {} {}/{} 1 {}/$SLURM_JOB_ID/{}_{}_{}_ {} --minsnaps {} --maxsnaps {} --pus {} &\n".format(typ, inputdir, fn(graph), outputdir, logfn(graph), typ, dupid, thins, minsnaps, maxsnaps, pus))
                writer.write("wait\n")
"""
            for graphs in zip(Jobgraphs[::4], Jobgraphs[1::4], Jobgraphs[2::4], Jobgraphs[3::4]):
                writer.write("./autocorrelation_realworld {} {}/{} 10 {}/$SLURM_JOB_ID/{}_{}_ {} --minsnaps {} --maxsnaps {} --pus {} &\n".format(typ, inputdir, fn(graphs[0]), outputdir, logfn(graphs[0]), typ, thins, minsnaps, maxsnaps, pus))
                writer.write("./autocorrelation_realworld {} {}/{} 10 {}/$SLURM_JOB_ID/{}_{}_ {} --minsnaps {} --maxsnaps {} --pus {} &\n".format(typ, inputdir, fn(graphs[1]), outputdir, logfn(graphs[1]), typ, thins, minsnaps, maxsnaps, pus))
                writer.write("./autocorrelation_realworld {} {}/{} 10 {}/$SLURM_JOB_ID/{}_{}_ {} --minsnaps {} --maxsnaps {} --pus {} &\n".format(typ, inputdir, fn(graphs[2]), outputdir, logfn(graphs[2]), typ, thins, minsnaps, maxsnaps, pus))
                writer.write("./autocorrelation_realworld {} {}/{} 10 {}/$SLURM_JOB_ID/{}_{}_ {} --minsnaps {} --maxsnaps {} --pus {} &\n".format(typ, inputdir, fn(graphs[3]), outputdir, logfn(graphs[3]), typ, thins, minsnaps, maxsnaps, pus))
                writer.write("wait\n")
            for graph in Jobgraphs[4*(len(Jobgraphs)//4):]:
                writer.write("./autocorrelation_realworld {} {}/{} 10 {}/$SLURM_JOB_ID/{}_{}_ {} --minsnaps {} --maxsnaps {} --pus {} &\n".format(typ, inputdir, fn(graph), outputdir, logfn(graph), typ, thins, minsnaps, maxsnaps, pus))
"""
