models=(`find ../train/trained_models/ -name "*json"`)

##################################
# Solid Lines + Act. Prec. Table #
##################################
for model in "${models[@]}"; do
    env=`echo $model | sed s/_checkpoint.json//g | sed 's/\.\.\/train\/trained_models\///g'`
    outfile_table=results_table_$env

    for W in 32 16 8 4 1; do
	echo "python evaluate.py --method cutlass --model_path $model --env $env --W_bits_1 $W --W_bits_2 $W --W_bits_3 $W --A_bits_1 $W --A_bits_2 $W --A_bits_3 $W >> $outfile_table"
	python evaluate.py --method cutlass --model_path $model --env $env --W_bits_1 $W --W_bits_2 $W --W_bits_3 $W --A_bits_1 $W --A_bits_2 $W --A_bits_3 $W >> $outfile_table
    done
    for W in 16 8 7 6 5 4 3 2 1; do
	for A in 32 16 8; do
	    echo "python evaluate.py --method pbatch --model_path $model --env $env --W_bits_1 $W --W_bits_2 $W --W_bits_3 $W --A_bits_1 $A --A_bits_2 $A --A_bits_3 $A >> $outfile_table"
	    python evaluate.py --method pbatch --model_path $model --env $env --W_bits_1 $W --W_bits_2 $W --W_bits_3 $W --A_bits_1 $A --A_bits_2 $A --A_bits_3 $A >> $outfile_table
	done
    done
done
