#! /bin/bash
# Script to change the CUDA version
VERSION=$1
if [ $VERSION == "11.2" ]
then
        OLD_VERSION="12.3"
        CUPY="11x"
        OLD_CUPY="12x"
elif [ $VERSION=="12.3" ]
then
        OLD_VERSION="11.2"
        CUPY="12x"
        OLD_CUPY="11x"
else
        echo "Undefined version ${VERSION}"
        exit 1
fi

module unload "cuda${OLD_VERSION}/toolkit"
module load "cuda${VERSION}/toolkit"
module list

# CuPy has CUDA version specific packages; first uninstall the old one before installing the new one
pip uninstall --yes "cupy-cuda${OLD_CUPY}"
pip install --force-reinstall --ignore-installed "cupy-cuda${CUPY}"

# force recompile PyCUDA, as on install it compiles against the current CUDA version
pip install --force-reinstall --ignore-installed --no-cache-dir --no-binary :all: pycuda
