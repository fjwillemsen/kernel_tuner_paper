""""Run the tuning of each combination of kernel and platform for each searchspace constructor."""

from pathlib import Path

from kernels.hotspot.hotspot import tune as tune_hotspot

kernels = ["hotspot"]           # names of the kernel and folder in the kernels folder (must be the same)
platforms = [("CUDA", "A4000")] # tuple of language and device, for language choose from CUDA, HIP and OpenCL
iterations = 10                 # number of times to repeat each tuning run
searchspace_constructors = [    # the searchspace construction frameworks to use
    "pythonconstraint",
    "pyatf",
    "bruteforce",
]

# execute the tuning for each combination
for iteration in range(iterations):
    for searchspace_constructor in searchspace_constructors:
        for kernel in kernels:
            if len(kernels) > 1:
                print(f"running kernel {kernel}:")
            for language, device in platforms:
                if len(platforms) > 1:
                    print(f"  on {device=} with {language=}:")
                # create the cache path and skip this combination if it already exists
                cachefile_path = Path(f"results/hotspot/{device.upper()}_f={searchspace_constructor}_i={iteration}.json")
                if cachefile_path.exists():
                    print(f"  skipping {searchspace_constructor} (iter. {iteration}) as it already exists")
                    continue

                # set the tuning parameters
                strategy_options = {
                    'time_limit': 20*60,     # time limit in seconds
                    'searchspace_construction_options': {
                        'framework': searchspace_constructor
                    }
                }

                # tune
                res, env = tune_hotspot(
                    device_name=device, 
                    strategy="random_sample", 
                    strategy_options=strategy_options,
                    simulation_mode=False,
                    lang=language,
                    verbose=False,
                    cachefile_path=str(cachefile_path)
                )
                print(f"{searchspace_constructor} (iter. {iteration}) evaluated {len(res)} configs in ~20 minutes")

                # assert the cache file path now exists
                assert cachefile_path.exists(), f"Cachefile does not exist at {cachefile_path}"

    print("-------------------------")
