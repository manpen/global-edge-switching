boilerplate = "" \
+ "#!/bin/bash\n" \
+ "#SBATCH --job-name=autocorr\n" \
+ "#SBATCH --partition=general1\n" \
+ "#SBATCH --ntasks=1\n" \
+ "#SBATCH --cpus-per-task=16\n" \
+ "#SBATCH --mem-per-cpu=3500\n" \
+ "#SBATCH --time=48:00:00\n" \
+ "#SBATCH --no-requeue\n" \
+ "#SBATCH --mail-type=FAIL\n" \
+ "\n" \
+ "DIR=\"/scratch/memhierarchy/penschuck/networks/network-repository.com/\"\n" \
+ "OUTPUTDIR=\"/scratch/memhierarchy/tran/autocorrelation/$SLURM_JOB_ID/\"\n" \
+ "mkdir -p $OUTPUTDIR\n" \
+ "cp /home/memhierarchy/tran/edge-switching/release/autocorrelation_realworld .\n"

inputdir = "/scratch/memhierarchy/penschuck/networks/network-repository.com"
outputdir = "/scratch/memhierarchy/tran/autocorrelation"
jobs = 20
minsnaps = 400
maxsnaps = 400
pus = 16
L = []
with open("sorted.csv", "r") as reader:
    for line in reader.read().splitlines()[1::]:
        s = line.split(",")
        if int(s[2]) >= 100 and int(s[1]) <= 100000 and int(s[2]) <= 5000000:
            L.append((s[0], int(s[1]), int(s[2])))
    
L.sort(key=lambda x: x[2])
total_edges = sum(map(lambda x: x[2], L))
edges_by_jobs = total_edges / jobs

J = [[] for i in range(jobs)]
current_sum = 0
index = 0
for t in L:
    current_sum += t[2]
    if current_sum > edges_by_jobs:
        index = index + 1
        current_sum = 0
    J[index].append(t)

for Jobgraphs, index in zip(J, range(len(J))):
    with open("job-{}.sh".format(index), "w") as writer:
        writer.write(boilerplate)
        for graph in Jobgraphs:
            log_fn = graph[0].split("/")[-1].split(".simp-undir-edges")[0]
            writer.write("./autocorrelation_realworld 1 {}/{} 20 1 2 3 4 5 6 8 9 10 12 14 15 16 18 20 30 40 50 --minsnaps {} --maxsnaps {} --pus {} >> {}/$SLURM_JOB_ID/{}-1.log\n".format(inputdir, graph[0], minsnaps, maxsnaps, pus, outputdir, log_fn))
            writer.write("./autocorrelation_realworld 2 {}/{} 20 1 2 3 4 5 6 8 9 10 12 14 15 16 18 20 30 40 50 --minsnaps {} --maxsnaps {} --pus {} >> {}/$SLURM_JOB_ID/{}-2.log\n".format(inputdir, graph[0], minsnaps, maxsnaps, pus, outputdir, log_fn))
