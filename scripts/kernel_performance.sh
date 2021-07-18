# Benchmark kernel performance. Reproduce table in paper. 

sizes=( 512 1024 2048 4096 )
w_bits=( 1 2 4 8 )
a_bits=( 8 16 32 )

bits_cutlass=( 16 8 4 1 )

for s in ${sizes[@]}; do

    baseline=`python src/pytorch/fake/benchmark_performance.py $s $s cutlass 32 32`

    # Cutlass
    for b in ${bits_cutlass[@]}; do
	t=`python src/pytorch/fake/benchmark_performance.py $s $s cutlass $b $b`
	res=`echo "scale=2 ; $baseline / $t" | bc`
	echo "Matrix size $s x $s, Cutlass bits=$b - $res"
    done

    # Pbatch
    for w in ${w_bits[@]}; do
	for a in ${a_bits[@]}; do
	    t=`python src/pytorch/fake/benchmark_performance.py $s $s pbatch $w $a`
	    res=`echo "scale=2 ; $baseline / $t" | bc`
	    echo "Matrix size $s x $s, Pbatch-$w (a=$a) - $res"
	done
    done
    
done
