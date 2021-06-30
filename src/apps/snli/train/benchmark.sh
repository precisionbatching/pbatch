W_bits=( 1 2 3 4 8 32 )
A_bits=( 1 2 4 8 16 32 )

python3 -m spacy download en

for W in "${W_bits[@]}"; do
    for A in "${A_bits[@]}"; do
        echo "TRAINING WITH W=$W, A=$A"
        echo "------------------------"
        python3 snli_train.py --W1_bits $W --A1_bits $A --W2_bits $W --A2_bits $A --logfile "savedata/logdata/weightbits=${W},actbits=${A}_out" --model_dir "savedata/models/" --gpu 1
    done
done
