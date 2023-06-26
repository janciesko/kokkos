#/bin/bash
BENCHMARK=$1
DATA_SIZE=33554432 #256MB

DS=$DATA_SIZE
echo "elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.res
for datasize in $(seq 1 10); do
    for reps in $(seq 1 10); do
	#echo "$datasize,$reps,$DS"
	./$BENCHMARK 8192 $DS 10 0 | tee -a $BENCHMARK.res 
    done	
  let DS=$DS+33554432
done

let DS=$DATA_SIZE
for datasize in $(seq 1 10); do
    for reps in $(seq 1 10); do
        #echo "$datasize,$reps,$DS"
        ./$BENCHMARK 8192 $DS 10 1 | tee -a $BENCHMARK.res
    done
  let DS=$DS+33554432
done
