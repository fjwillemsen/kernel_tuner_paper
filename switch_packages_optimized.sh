pip uninstall --yes kernel_tuner
pip uninstall --yes python-constraint
pip install --force-reinstall --ignore-installed --no-cache-dir python-constraint2
pip install -e ../kernel_tuner
