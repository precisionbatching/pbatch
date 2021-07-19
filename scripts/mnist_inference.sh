
outfile="mnist_out"
rm -f $outfile

# PBatch
W_bits=( 1 4 8 )
A_bits=( 8 16 32 )
for w in ${W_bits[@]}; do
    for a in ${A_bits[@]}; do
	python src/apps/mnist/inference/inference.py --model_path src/apps/mnist/inference/mnist_model.ckpt  --method pbatch --W_bits $w --A_bits $a >> $outfile	
    done
done

# Cutlass
cutlass_bits=( 32 16 8 4 1 )
for b in ${cutlass_bits[@]}; do
    python src/apps/mnist/inference/inference.py --model_path src/apps/mnist/inference/mnist_model.ckpt  --method cutlass --W_bits $b --A_bits $b >> $outfile
done

python src/plotting/plot_end2end.py $outfile mnist
