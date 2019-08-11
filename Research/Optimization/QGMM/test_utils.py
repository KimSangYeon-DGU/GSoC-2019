import json, os

test_cases = []
'''
test_cases.append({"name": "t_lambda_impact_2",
                    "mean1":[3.427976229515216, 61.46413393088303], 
                    "mean2":[4.5517041217554945, 51.756595162050985],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False,
					"data":"faithful.csv"})

test_cases.append({"name": "t_lambda_impact_3",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False,
					"data":"faithful.csv"})

test_cases.append({"name": "t_lambda_impact_5",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False,
					"data":"faithful.csv"})

test_cases.append({"name": "t_lambda_impact_2",
                    "mean1":[3.427976229515216, 61.46413393088303], 
                    "mean2":[4.5517041217554945, 51.756595162050985],
                    "ld":500,
                    "phis":[45, -45],
                    "run":False,
					"data":"faithful.csv"})

test_cases.append({"name": "t_lambda_impact_3",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":500,
                    "phis":[45, -45],
                    "run":False,
					"data":"faithful.csv"})


test_cases.append({"name": "t_lambda_impact_5",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":500,
                    "phis":[45, -45],
                    "run":False,
					"data":"faithful.csv"})
'''

test_cases.append({"name": "t_multiple_1",
                    "mean1":[-0.972979456933269, -5.7560544338372392], 
                    "mean2":[2.686047548070453, -1.1307013280230782],
                    "mean3":[3.221786450205375, 8.262448613633105],
                    "mean4":[-3.3977125108429398, 5.4590524145870889],
                    "mean5":[7.307847427018068, -4.354851796948616],
                    "ld":500,
                    "phis":[0, 0, 0, 0, 0],
                    "run":False,
					"data":"multiple_5_2.csv"})


for test_case in test_cases:
	test_name = "{0}_{1}_{2}".format(test_case["name"], \
		int(test_case["phis"][0]) - int(test_case["phis"][1]), test_case["ld"])
	
	file_path = "jsons/{0}".format(test_name)

	if os.path.exists(file_path) == False:
		os.mkdir(file_path)
    
	with open("jsons/{0}/{0}.json".format(test_name), 'w') as outfile:
		json.dump(test_case, outfile)