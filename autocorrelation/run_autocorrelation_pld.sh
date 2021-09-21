#!/usr/bin/sh

for i in $(seq 7 13);
do
    for g in "3" "2.5" "2.2" "2.1" "2.01";
    do
        for k in $(seq 1 40);
        do
            ../release/autocorrelation_pld 1 1 $((2**i)) ${g} 1 2 3 4 5 6 7 8 9 10 12 14 15 16 18 20 21 24 25 26 27 28 30 --minsnaps 10000 --maxsnaps 10000 --pus 16 >> output_pld_2p${i}_g${g}-1.log;
            ../release/autocorrelation_pld 2 1 $((2**i)) ${g} 1 2 3 4 5 6 7 8 9 10 12 14 15 16 18 20 21 24 25 26 27 28 30 --minsnaps 10000 --maxsnaps 10000 --pus 16 >> output_pld_2p${i}_g${g}-2.log;
        done
    done
done
