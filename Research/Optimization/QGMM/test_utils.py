import json, os

test_cases = []

'''
test_cases.append({"name": "t_validity_1",
                    "means":[[2.6585135519388348, 54.66062219876824], 
                            [3.1085745233652995, 77.99698134521407]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})

test_cases.append({"name": "t_validity_2",
                    "means":[[3.427976229515216, 61.46413393088303], 
                             [4.5517041217554945, 51.756595162050985]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})

test_cases.append({"name": "t_validity_3",
                    "means":[[2.756031811312966, 76.62447648112042], 
                             [2.9226572802266397, 88.3509418943818]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})
                    
test_cases.append({"name": "t_validity_4",
                    "means":[[4.893025788130122, 59.46713813379837], 
                             [2.080000263954121, 78.15976694366192]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})

test_cases.append({"name": "t_validity_5",
                    "means":[[4.171021823127277, 83.66322004888708],
                             [1.781079954983019, 95.411542531776]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})

test_cases.append({"name": "t_validity_6",
                    "means":[[4.171021823127277, 43.66322004888708],
                             [1.781079954983019, 95.411542531776]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})
'''
'''
test_cases.append({"name": "t_validity_7",
                    "means":[[4.171021823127277, 53.66322004888708],
                             [1.781079954983019, 95.411542531776]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
                    "data":"faithful.csv"})
                    
test_cases.append({"name": "t_validity_8",
                    "means":[[6.171021823127277, 83.66322004888708],
                             [7.281079954983019, 95.411542531776]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})

'''
'''
test_cases.append({"name": "t_validity_9",
                    "means":[[6.171021823127277, 83.66322004888708],
                             [3.581079954983019, 45.411542531776]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, 45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})
'''
'''
test_cases.append({"name": "t_multiple_1",
                    "means":[[-0.972979456933269, -5.7560544338372392],
                             [2.686047548070453, -1.1307013280230782],
                             [3.221786450205375, 8.262448613633105],
                             [-3.3977125108429398, 5.4590524145870889],
                             [7.307847427018068, -4.354851796948616]],
                    "covs":[[[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]],
                            [[0.5, 0.0], [0.0, 0.5]]],
                    "alphas":[0.5, 0.5, 0.5, 0.5, 0.5],
                    "phis":[45, 45, 45, 45, 45],
                    "gaussians":5,
                    "dimensionality":2,
                    "ld":500,
                    "data":"multiple_5_2.csv"})
'''

test_cases.append({"name": "t_fast_5",
                    "means":[[4.171021823127277, 83.66322004888708],
                             [1.781079954983019, 95.411542531776]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[45, -45],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":1,
					          "data":"faithful.csv"})

for test_case in test_cases:
	test_name = "{0}_{1}_{2}".format(test_case["name"], \
		int(test_case["phis"][0]) - int(test_case["phis"][1]), test_case["ld"])
	
	file_path = "jsons/{0}".format(test_name)

	if os.path.exists(file_path) == False:
		os.mkdir(file_path)
    
	with open("jsons/{0}/{0}.json".format(test_name), 'w') as outfile:
		json.dump(test_case, outfile)