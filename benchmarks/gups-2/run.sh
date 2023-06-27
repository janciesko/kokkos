#/bin/bash
BENCHMARK=$1
DATA_SIZE=1024 #64kB

DS=$DATA_SIZE
echo "elems,size,idx,idxsize,isAtomic,GUPS" | tee $BENCHMARK.res
for datasize in $(seq 1 20); do
    for reps in $(seq 1 5); do
	#echo "$datasize,$reps,$DS"
	./$BENCHMARK 8192 $DS 7 0 | tee -a $BENCHMARK.res 
    done	
  let DS=$DS*2
done

let DS=$DATA_SIZE
for datasize in $(seq 1 20); do
    for reps in $(seq 1 5); do
        #echo "$datasize,$reps,$DS"
        ./$BENCHMARK 8192 $DS 7 1 | tee -a $BENCHMARK.res
    done
  let DS=$DS*2
done
