W_bits=( 1 2 3 4 8 32 )
A_bits=( 1 2 4 8 16 32 )

python mnist_train.py --W_bits 1 --A_bits 1 --logfile "benchmark_data/logdata/W=1,A=1" --model_dir "benchmark_data/savedata/W=1,A=1"
python mnist_train.py --W_bits 1 --A_bits 32 --logfile "benchmark_data/logdata/W=1,A=32" --model_dir "benchmark_data/savedata/W=1,A=32"

python mnist_train.py --W_bits 32 --A_bits 32 --logfile "benchmark_data/logdata/W=32,A=32" --model_dir "benchmark_data/savedata/W=32,A=32"

python mnist_train.py --W_bits 8 --A_bits 8 --logfile "benchmark_data/logdata/W=8,A=8" --model_dir "benchmark_data/savedata/W=8,A=8"
python mnist_train.py --W_bits 8 --A_bits 32 --logfile "benchmark_data/logdata/W=8,A=32" --model_dir "benchmark_data/savedata/W=8,A=32"

python mnist_train.py --W_bits 4 --A_bits 4 --logfile "benchmark_data/logdata/W=4,A=4" --model_dir "benchmark_data/savedata/W=4,A=4"
python mnist_train.py --W_bits 4 --A_bits 32 --logfile "benchmark_data/logdata/W=4,A=32" --model_dir "benchmark_data/savedata/W=4,A=32"

