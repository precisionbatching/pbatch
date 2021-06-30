
gitrootdir=`git rev-parse --show-toplevel`
echo "Git Root Directory: $gitrootdir"
export PYTHONPATH=$gitrootdir/src/pytorch/fake:$PYTHONPATH
export PYTHONPATH=$gitrootdir/src/pytorch/cutlass_linear:$PYTHONPATH
export PYTHONPATH=$gitrootdir/src/pytorch/pbatch:$PYTHONPATH

