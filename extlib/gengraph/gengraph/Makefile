CC = g++ -Wall -O3

SRC = 
INCLUDE = include/
BIN = bin/

OBJ =$(BIN)random.o $(BIN)powerlaw.o $(BIN)degree_sequence.o $(BIN)box_list.o $(BIN)graph_molloy_hash.o $(BIN)graph_molloy_optimized.o

.cpp.o :
	$(CC) -c -o $(BIN)$@ $(INCLUDE)$<

all :
	make obj
	make graph
	make distrib

obj :
	$(CC) -c -o $(BIN)graph_molloy_hash.o $(INCLUDE)graph_molloy_hash.cpp
	$(CC) -c -o $(BIN)graph_molloy_optimized.o $(INCLUDE)graph_molloy_optimized.cpp
	$(CC) -c -o $(BIN)degree_sequence.o $(INCLUDE)degree_sequence.cpp
	$(CC) -c -o $(BIN)powerlaw.o $(INCLUDE)powerlaw.cpp
	$(CC) -c -o $(BIN)random.o $(INCLUDE)random.cpp
	$(CC) -c -o $(BIN)box_list.o $(INCLUDE)box_list.cpp

clean :
	rm -f $(BIN)* $(SRC)~* $(INCLUDE)~* $(SRC)*~ $(INCLUDE)*~

graph : $(OBJ)
	$(CC) $(OBJ) $(SRC)mr-connected.cpp -O3 -o $(BIN)graph

distrib : $(OBJ)
	$(CC) $(OBJ) $(SRC)distrib.cpp -O3 -o $(BIN)distrib

