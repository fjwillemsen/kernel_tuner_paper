pip uninstall --yes kernel-tuner
pip uninstall --yes python-constraint2
pip install --force-reinstall --ignore-installed --no-cache-dir --no-binary :all:  python-constraint
pip install --force-reinstall --ignore-installed --no-cache-dir kernel-tuner==0.4.4
