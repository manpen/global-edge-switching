COPYRIGHT:
see file GNU_GPL.txt

COMPILING:
To compile the whole package, just enter the directory and type 'make'.
If you don't have the 'make' utility, try
$ g++ -Wall -O3 include/*.cpp mr-connected.cpp -o bin/graph
$ g++ -Wall -O3 include/*.cpp distrib .cpp -o bin/distrib

SHORT DESCRIPTION:
"bin/distrib n alpha min max" generates a sample of n integers taken in [min,max] from a
power-law distribution of exponent alpha.
"bin/distrib n alpha min max z" generates a sample of n integers taken in [min,max] from a
heavy-tailed distribution of exponent alpha and average z.
"bin/graph filename" reads the degree distribution given in file "filename" and generates
 a random simple connected graph with this degree sequence.
 
EXAMPLE:
$ bin/distrib 10000 2.5 1 1000 5 > degs
$ bin/graph degs > mygraph

EXAMPLE #2:
$ bin/distrib -v 10000 2.5 1 1000 5 | bin/graph -vv  > mygraph

HELP:
You can get some short help with bad arguments or with the -? option, like
$ bin/distrib -?
$ bin/graph -?

PLAYING:
To compare the computation time with the Gkantsidis heuristics:
$ bin/distrib -v 10000 6 2.5 > degrees
$ bin/graph -vv -t degrees > mygraph
$ bin/graph -vv -g -t degrees > mygraph

FORMAT:
distrib output format :
 each line is <degree> <nb_vertices_with_this_degree>
graph output format :
 each line is <vertex> <neighbour 1> <neighbour 2> ...

DISCLAIMER:
I do NOT warrantee that these programs will be harmless to your computer,
or will not cause any data loss. Use them at your own risk !
However, if you encounter any problem by using this software, feel
free to contact me (address below)

If you have any questions, remarks or comments are welcome, write to
 fabien [dot] viger [at] ens [dot] fr

November 10th, 2004
