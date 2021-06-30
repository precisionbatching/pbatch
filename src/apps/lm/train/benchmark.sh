W_bits=( 1 2 3 4 8 32 )
A_bits=( 1 2 4 8 16 32 )

for W in "${W_bits[@]}"; do
    for A in "${A_bits[@]}"; do
        echo "TRAINING WITH W=$W, A=$A"
        echo "------------------------"
        python main.py --W_bits $W --A_bits $A --logfile "savedata/logdata/weightbits=${W},actbits=${A}_out" --model_dir "savedata/models/" --cuda
    done
done
