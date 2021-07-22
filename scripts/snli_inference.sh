outfile="snli_out"
rm -f $outfile

# Install spacy en
python -m spacy download en

# Construct snli model file
cat src/apps/snli/inference/model/x* > src/apps/snli/inference/model/snli_model.pt

# PBatch
W_bits=( 1 4 8 )
A_bits=( 8 16 32 )
for w in ${W_bits[@]}; do
    for a in ${A_bits[@]}; do
	python src/apps/snli/inference/inference.py --model_path src/apps/snli/inference/model/snli_model.pt  --method pbatch --W_bits $w --A_bits $a >> $outfile
    done
done

# Cutlass
cutlass_bits=( 32 16 8 4 1 )
for b in ${cutlass_bits[@]}; do
    python src/apps/snli/inference/inference.py --model_path src/apps/snli/inference/model/snli_model.pt   --method cutlass --W_bits $b --A_bits $b >> $outfile
done

python src/plotting/plot_end2end.py $outfile snli 0
