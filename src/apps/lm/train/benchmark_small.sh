python main.py --W_bits 32 --A_bits 32 --logfile "benchmark_data/logdata/W=32,A=32" --model_dir "benchmark_data/savedata/W=32,A=32" --cuda > output_32_32 2>&1 &

#python main.py --W_bits 1 --A_bits 1 --logfile "benchmark_data/logdata/W=1,A=1" --model_dir "benchmark_data/savedata/W=1,A=1" --cuda > output_1_1 2>&1 &
python main.py --W_bits 1 --A_bits 32 --logfile "benchmark_data/logdata/W=1,A=32" --model_dir "benchmark_data/savedata/W=1,A=32" --cuda > output_1_32 2>&1 &

python main.py --W_bits 4 --A_bits 4 --logfile "benchmark_data/logdata/W=4,A=4" --model_dir "benchmark_data/savedata/W=4,A=4" --cuda > output_4_4 2>&1 &
#python main.py --W_bits 4 --A_bits 32 --logfile "benchmark_data/logdata/W=4,A=32" --model_dir "benchmark_data/savedata/W=4,A=32" --cuda > output_4_32 2>&1 &

