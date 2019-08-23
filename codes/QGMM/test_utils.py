import json, os

test_cases = []

test_cases.append({"name": "01_mix_1p5",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_1.5.csv"})

test_cases.append({"name": "01_mix_1p5",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, -45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_1.5.csv"})

test_cases.append({"name": "01_mix_2",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_2.csv"})

test_cases.append({"name": "01_mix_2",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, -45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_2.csv"})

test_cases.append({"name": "01_mix_2p5",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_2.5.csv"})

test_cases.append({"name": "01_mix_2p5",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, -45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_2.5.csv"})          

test_cases.append({"name": "01_mix_3",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_3.csv"})

test_cases.append({"name": "01_mix_3",
                    "means":[[-1.1431749105000186, 2.0529438865919479],
                             [1.846096025789632, -0.5656381542446731]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, -45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":0,
					          "data":"gmm_radius_3.csv"})     

for test_case in test_cases:
	test_name = "{0}_{1}_{2}".format(test_case["name"], \
		int(test_case["phis"][0]) - int(test_case["phis"][1]), test_case["ld"])
	
	file_path = "jsons/{0}".format(test_name)

	if os.path.exists(file_path) == False:
		os.mkdir(file_path)
    
	with open("jsons/{0}/{0}.json".format(test_name), 'w') as outfile:
		json.dump(test_case, outfile)