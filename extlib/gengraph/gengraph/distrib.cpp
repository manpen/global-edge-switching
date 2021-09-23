#include "include/header.h"

#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#include "include/degree_sequence.h"

static bool RAW = false;
static bool TEST_REALIZABILITY = true;
static bool TEST_CONNECTABILITY = true;
static bool FORCE_REALIZABILITY = false;
static bool FORCE_CONNECTABILITY = false;
static bool MONITOR_TIME = false;

static int n = 0;     // #vertices
static double alpha;  // exponent
static int mini;      // Min degree
static int maxi;      // Max degree (-1 if unlimited)
static double z=-1.0; // Average degree (-1 if unspecified)

void scan_args(int &, char** &);

//___________________________________________________________________________
int main(int argc, char** argv) {

#ifdef _WIN32
  clock_t __clock = clock();
  time_t __time;
  set_priority_low();
  my_srandom(__clock + time(&__time) + _getpid());
#else
  my_srandom(getpid());
#endif

  scan_args(argc, argv);
  for(;;) {
    degree_sequence dd(n,alpha,mini,maxi,z);
    if(z==-1.0 && VERBOSE()) fprintf(stderr,"Average degree obtained : %f\n",double(dd.sum())/double(n));

    // Adjust degree sequence parity
    if(VERBOSE()) fprintf(stderr,"Adjusting degree sequence parity...");
    dd.make_even(mini,maxi);
    if(VERBOSE()) fprintf(stderr,"done\n");

    // Check/Force connectability
    if(FORCE_CONNECTABILITY || TEST_CONNECTABILITY) {
      if(VERBOSE()) fprintf(stderr,"Checking Connectability...");
      if(dd.sum()/2<dd.size()-1) {
        if(VERBOSE()) fprintf(stderr,"failed\n");
        if(FORCE_CONNECTABILITY) continue; 
        fprintf(stderr,"Error : degree sequence is not connectable. Average degree = %f\n",double(dd.sum())/double(n));
        return 3;
      }
    }

    // Check/Force Realizability
    if(FORCE_REALIZABILITY || TEST_REALIZABILITY) {
      if(VERBOSE()) fprintf(stderr,"done\nChecking Realizability...");
      if(!dd.havelhakimi()) {
        if(VERBOSE()) fprintf(stderr,"failed\n");
	if(FORCE_REALIZABILITY) continue; 
        fprintf(stderr,"Error : degree sequence is not realizable\n");
        return 4;
      }
    }
    if(VERBOSE()) fprintf(stderr,"done\n");
 
    // If we're here, we succeeded. So, we should print & exit
    if(RAW) dd.print();
    else dd.print_cumul();
  
    if(MONITOR_TIME)
      fprintf(stderr,"Time used : %f\n",double(clock())/double(CLOCKS_PER_SEC));
    
    return 0;
  }
}


//___________________________________________________________________________
void scan_args(int &argc, char** &argv) {
  SET_VERBOSE(VERBOSE_NONE);
  bool HELP = false;
  int c = 1;
  for(int a=1; a<argc; a++) {
    if(strcmp(argv[a],"-v")==0) SET_VERBOSE(VERBOSE_SOME);
    else if(strcmp(argv[a],"-t")==0) MONITOR_TIME = true;
    else if(strcmp(argv[a],"-?")==0 || strcmp(argv[1],"--help")==0 || strcmp(argv[1],"/?")==0) HELP = true;
    else if(strcmp(argv[a],"-s")==0) my_srandom(0);
    else if(strcmp(argv[a],"-r")==0) FORCE_REALIZABILITY = true;
    else if(strcmp(argv[a],"-c")==0) FORCE_CONNECTABILITY = true;
    else if(strcmp(argv[a],"-nr")==0) TEST_REALIZABILITY = false;
    else if(strcmp(argv[a],"-nc")==0) TEST_CONNECTABILITY = false;
    else if(strcmp(argv[a],"-raw")==0) RAW = true;
    else argv[c++]=argv[a]; // Arg. is not an option
  }
  argc = c;
  if(HELP || (argc!=5 && argc!=6)) {
    fprintf(stderr,"Usage : %s [options] <n> <exp> <min> <max> [<z>]\n",argv[0]);
    fprintf(stderr,"Generates either a binomial degree sequence of fixed average and size, or\n");
    fprintf(stderr,"a power-law degree sequence of size n, exponent exp, optional mean z\n");
    fprintf(stderr,"If <exp> is 0 the distribution is binomial. Then <z> MUST be specified\n");
    fprintf(stderr,"The sequence checks by default its realizability as a simple connected graph.\n");
    fprintf(stderr,"    [-nc] option skips the connectability test\n");
    fprintf(stderr,"    [-nr] option skips the realizability test\n");
    fprintf(stderr,"    [-c] option forces the sequence to be connectable\n");
    fprintf(stderr,"    [-r] option forces the sequence to be realizable\n");
    fprintf(stderr,"    [-v] option causes extra verbose\n");
    fprintf(stderr,"    [-t] option monitors computation time\n");
    fprintf(stderr,"    [-s] option does a srandom(0) to get a constant random sequence\n");
    fprintf(stderr,"    [-raw] option prints the degree sequence\n");
    fprintf(stderr,"    [-distrib] option prints the degree distribution {default}\n");
    fprintf(stderr,"* If <max>=-1, max=infinity is assumed.\n");
    fprintf(stderr,"* For power-law : If <z> is unspecified, the law will be P(X=k+mini)~(k)^(-exp)\n");
    fprintf(stderr,"* If <z> is specified, the law will be P(X=k)~(k+offset)^(-exp) instead,\n");
    fprintf(stderr,"  with 'offset' tuned to get the proper mean z.\n");
    fprintf(stderr,"  Moreover, the sequence will have a sum exactly equal to 2*floor(n*z/2)\n");
    exit(-1);
  }

  // Create degree distribution with power-law
  if(sscanf(argv[1],"%d",&n)!=1) {
    fprintf(stderr,"Error : first argument must be an integer (#vertices)\n");
    exit(1);
  }
  if(sscanf(argv[2],"%lf",&alpha)!=1) {
    fprintf(stderr,"Error : 2nd argument must be a double (exponent)\n");
    exit(2);
  }
  if(sscanf(argv[3],"%d",&mini)!=1) {
    fprintf(stderr,"Error : 3rd argument must be an integer (min degree)\n");
    exit(3);
  }
  if(sscanf(argv[4],"%d",&maxi)!=1) {
    fprintf(stderr,"Error : 4th argument must be an integer (max degree)\n");
    exit(4);
  }
  if(mini>maxi && maxi>=0) {
    fprintf(stderr,"Error : degmin > degmax\n");
    exit(4);
  }
  if(argc!=6 && z==0.0) {
    fprintf(stderr,"Error: when z=0, the distribution is binomial, and needs an average <z>\n");
    exit(5);
  }
  if(argc==6 && (sscanf(argv[5],"%lf",&z)!=1 || z<=double(mini) || (z>=0.5*double(mini+maxi) && maxi>=0))) {
    fprintf(stderr,"Error : 5th arg. (=mean) must be a double in ]min, (min+max)/2[ = ]%d,%f[\n",mini,0.5*double(mini+maxi));
    exit(5);
  }
}

