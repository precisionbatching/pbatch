
outfile="outfile"
rm -f $outfile

# Construct lm model file
cat src/apps/lm/inference/model/x* > src/apps/lm/inference/model/lm_model.pt

# PBatch
W_bits=( 1 4 8 )
A_bits=( 8 16 32 )
for w in ${W_bits[@]}; do
    for a in ${A_bits[@]}; do
	python src/apps/lm/inference/inference.py --save src/apps/lm/inference/model/lm_model.pt --data src/apps/lm//train/wikitext-2/  --method pbatch --W_bits $w --A_bits $a >> $outfile
    done
done

# Cutlass
cutlass_bits=( 32 16 8 4 1 )
for b in ${cutlass_bits[@]}; do
    python src/apps/lm/inference/inference.py --save src/apps/lm/inference/model/lm_model.pt  --data src/apps/lm//train/wikitext-2/ --method cutlass --W_bits $b --A_bits $b >> $outfile
done

python src/plotting/plot_end2end.py $outfile lm 1
