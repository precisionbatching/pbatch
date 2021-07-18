# Benchmark kernel performance. Reproduce table in paper. 

sizes=( 512 1024 2048 4096 )
w_bits=( 1 2 4 8 )
a_bits=( 8 16 32 )

bits_cutlass=( 1 4 8 16 )

for s in ${sizes[@]}; do

    # Cutlass
    for b in $bits_cutlass; do
	t=`python src/pytorch/fake/benchmark_performance.py $s $s cutlass $b $b`
	echo "Matrix size $s x $s, Cutlass-$w bits=$b - $t"	
    done

    # Pbatch
    for w in ${w_bits[@]}; do
	for a in ${a_bits[@]}; do
	    t=`python src/pytorch/fake/benchmark_performance.py $s $s pbatch $w $a`
	    echo "Matrix size $s x $s, Pbatch-$w (a=$a) - $t"
	done
    done
    
done
