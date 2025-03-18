# Software repository for an upcoming Kernel Tuner paper

## Steps to reproduce results:
To start, make sure the kernel tuner and autotuning_methodology repositories and submodules have been correctly installed. 
To reproduce figures 2 and 3, run `python hypertuning_analysis.py`. Figures will appear in directory. 
To reproduce figure 6, run `python time_for_hyperparamtuning.py`. Figures will appear in directory. 
To reproduce figures 4 and 5, there are three levels:
1. Reproduce figures using the methodology result folder (`hyperparametertuning_milo`): put this at the root of `autotuning_methodology` and run ` autotuning_visualize experiment_files/compare_hypertuners_paper.json`. The figures will appear in `hyperparametertuning_milo/generated_graphs`.
2. Re-run comparison: run ` autotuning_visualize experiment_files/compare_hypertuners_paper.json` (this will take a long time!). The figures will appear in `hyperparametertuning_milo/generated_graphs`.
3. Re-run the hyperparameter tuning yourself using `python hyper.py [optimization algorithm]` (this will take a long time!). Parallel execution of different optimization algorithms is possible. See `kernel_tuner/kernel_tuner/hyper.py`. After completion, go to step 2.


Some of the output files are too large (>50MB) to be properly stored in Git. In these cases, the files have been compressed with `gzip -k [file].json`, and can be decompressed with `gzip -d -k [file].json.gz`.
