source setup.sh

# Install pbatch + cutlass baseline
pushd src/pytorch/cpp/pbatch/; bash install.sh; popd
pushd src/pytorch/cpp/cutlass_linear/; bash install.sh; popd
