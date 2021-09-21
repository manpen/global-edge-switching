import math
import pprint

def return_boilerplate(jobid, t, edges):
    approxtime = math.ceil(120. * float(edges) / 4000000.) + 3 
    boilerplate = "" \
        + "#!/bin/bash\n" \
        + "#SBATCH --job-name=ac-12e4-25e5-{}\n".format(2 * jobid + t - 1) \
        + "#SBATCH --partition=general1\n" \
        + "#SBATCH --ntasks=1\n" \
        + "#SBATCH --cpus-per-task=16\n" \
        + "#SBATCH --mem-per-cpu=3500\n" \
        + "#SBATCH --time={}:00:00\n".format(approxtime) \
        + "#SBATCH --no-requeue\n" \
        + "#SBATCH --mail-type=FAIL\n" \
        + "\n" \
        + "DIR=\"/scratch/memhierarchy/penschuck/networks/network-repository.com/\"\n" \
        + "OUTPUTDIR=\"/scratch/memhierarchy/tran/autocorrelation/$SLURM_JOB_ID/\"\n" \
        + "mkdir -p $OUTPUTDIR\n" \
        + "cp /home/memhierarchy/tran/edge-switching/release/autocorrelation_realworld .\n"
    return boilerplate

types = ["bio", "bn", "ca", "cit", "eco", "econ", "email", "heter", "ia", "inf", 
         "labeled", "massive", "power", "protein", "proximity", "rec", 
         "retweet_graphs", "road", "sc", "soc", "socfb", "tech", "web"]

inputdir = "/scratch/memhierarchy/penschuck/networks/network-repository.com"
outputdir = "/scratch/memhierarchy/tran/autocorrelation"
jobs = 15
minsnaps = 10000
maxsnaps = 10000
pus = 40
L = []

with open("sorted.csv", "r") as reader:
    for line in reader.read().splitlines()[1::]:
        s = line.split(",")
        n = int(s[1])
        m = int(s[2])
        stype = s[0].split("/")[0]
        if not (stype in types):
            continue
        # do 100k for now
        if m > 100000:
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

# 1 2 3 4 5 6 7 8 9 10 12 14 15 18 20 21 24 25 26 28 30 50 70 100
for Jobgraphs, index in zip(J, range(len(J))):
    if len(Jobgraphs) == 0:
        continue   
    jsum = sum(map(lambda k:k[2], Jobgraphs))
    for typ in [1, 2]:
        with open("job-{}.sh".format(2 * index + typ - 1), "w") as writer:
            writer.write(return_boilerplate(index, typ, jsum))
            for graph in Jobgraphs:
                log_fn = graph[0].split("/")[-1].split(".simp-undir-edges")[0]
                writer.write("./autocorrelation_realworld {} {}/{} {}/{}_ 40 1 2 3 4 5 6 8 9 10 12 14 15 18 20 21 24 25 26 28 30 50 70 100 --minsnaps {} --maxsnaps {} --pus {}\n".format(typ, inputdir, graph[0], outputdir, graph[0], minsnaps, maxsnaps, pus, outputdir, log_fn, typ))
