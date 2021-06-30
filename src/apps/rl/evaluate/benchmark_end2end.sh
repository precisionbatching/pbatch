models=(`find ../train/trained_models/ -name "*json" | grep -v reacher`)

##################################
# Solid Lines + Act. Prec. Table #
##################################
for model in "${models[@]}"; do
    env=`echo $model | sed s/_checkpoint.json//g | sed 's/\.\.\/train\/trained_models\///g'`
    outfile_table=end2end_table_$env

    # Cutlass
    for W1 in 8 4 1; do
	for W2 in 8 4 1; do
	    for W3 in 8 4 1; do
		echo "python evaluate.py --method cutlass --model_path $model --env $env --W_bits_1 $W1 --W_bits_2 $W2 --W_bits_3 $W3 --A_bits_1 $W1 --A_bits_2 $W2 --A_bits_3 $W3 >> $outfile_table"
		python evaluate.py --method cutlass --model_path $model --env $env --W_bits_1 $W1 --W_bits_2 $W2 --W_bits_3 $W3 --A_bits_1 $W1 --A_bits_2 $W2 --A_bits_3 $W3 >> $outfile_table
	    done
	done
    done

    # PBatch
    Ws=( 8 4 1 )
    As=( 16 8 )
    for W1 in "${Ws[@]}"; do
	for A1 in "${As[@]}"; do
	    for W2 in "${Ws[@]}"; do
		for A2 in "${As[@]}"; do
		    for W3 in "${Ws[@]}"; do
			for A3 in "${As[@]}"; do	    			    
			    echo "python evaluate.py --method pbatch --model_path $model --env $env --W_bits_1 $W1 --W_bits_2 $W2 --W_bits_3 $W3 --A_bits_1 $A1 --A_bits_2 $A2 --A_bits_3 $A3 >> $outfile_table"
			    python evaluate.py --method pbatch --model_path $model --env $env --W_bits_1 $W1 --W_bits_2 $W2 --W_bits_3 $W3 --A_bits_1 $A1 --A_bits_2 $A2 --A_bits_3 $A3 >> $outfile_table
			done
		    done
		done
	    done
	done
    done
done
