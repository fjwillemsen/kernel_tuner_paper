This folder contains Python bindings for ATF. 
The Python bindings are made using `pybind11`. 

Installation: 
1. Iniatiate the two submodules by `cd`ing into their respective folders and applying `git submodule update --init`. Afterwards, `cd` back to this folder. 
2. Make sure CMake is installed on your system or install it. 
3. `cd` into `extern/pybind11`. Do `mkdir build` and `cd build`.
4. Now run `cmake ..` followed by `make check -j 4`. This will make, compile and test the pybind11 installation. 

Building:
Given the [example application of the documentation](https://pybind11.readthedocs.io/en/latest/basics.html#creating-bindings-for-a-simple-function) we now attempt to import it in Python. 
For this we can compile it using: `c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3-config --includes) -I extern/pybind11/include example.cpp -o example$(python3-config --extension-suffix)`. With multiple Python versions installed, it may be necessary to change `python3-config` to e.g. `python3.9-config` if you are unable to import the module later (this can be checked by running `python3-config --extension-suffix`: if the resulting Python version differs from the target Python version). For macOS, change `-fPIC` to `-undefined dynamic_lookup` to avoid a `missing symbols` error. 

Running:
With the example application compiled, we can test it by opening a python interpreter and running `import example`, followed by `example.add(1, 3)`. This should return 4. 
