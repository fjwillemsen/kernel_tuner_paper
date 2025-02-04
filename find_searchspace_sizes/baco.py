# calculate the averages and percentages of the fifteen search spaces in the BaCO paper

spaces = {
    "TACO": {
        "cartesian": [15000000, 520000000000, 520000000000, 15000000, 1500000],
        "configs": [3000000, 47000, 78000, 6000000, 680000],
    },
    "RISE & ELEVATE": {
        "cartesian": [10000000, 110000000000, 1200000, 39000000, 14000, 7700000000, 14000],
        "configs": [29000, 150000000, 63000, 4200000, 3600, 10000000, 3600],
    },
    "HPVM2FPGA": {
        "cartesian": [256, 840000, 15000],
        "configs": [256, 840000, 15000],
    },
}

# 

# HPVM2FPGA

for application_name, values in spaces.items():
    assert len(values["cartesian"]) == len(values["configs"])
    num_spaces = len(values["cartesian"])

    # calculate statistics per search space
    lst_cartesian = list()
    lst_configs = list()
    lst_percentages = list()
    for i in range(num_spaces):
        cartesian = values["cartesian"][i]
        configs = values["configs"][i]
        percentage = (configs / cartesian) * 100
        lst_cartesian.append(cartesian)
        lst_configs.append(configs)
        lst_percentages.append(percentage)
    
    # calculate averages
    avg_cartesian = sum(lst_cartesian) / num_spaces
    avg_configs = sum(lst_configs) / num_spaces
    avg_percentage = sum(lst_percentages) / num_spaces

    # calculate standard deviations
    std_cartesian = (sum((x - avg_cartesian) ** 2 for x in lst_cartesian) / num_spaces) ** 0.5
    std_configs = (sum((x - avg_configs) ** 2 for x in lst_configs) / num_spaces) ** 0.5
    std_percentage = (sum((x - avg_percentage) ** 2 for x in lst_percentages) / num_spaces) ** 0.5

    # print statistics
    print(f"Statistics for {application_name}")
    print(f"Average number of Cartesian products: {round(avg_cartesian, 3)} ({round(std_cartesian, 3)})")
    print(f"Average number of configurations: {round(avg_configs, 3)} ({round(std_configs, 3)})")
    print(f"Average percentage of configurations: {round(avg_percentage, 3)} ({round(std_percentage, 3)})")
    print("")
