import json, os


test_cases = []
'''
## GMM basic test
test_cases.append({"name": "t_mix_1",
                    "mean1":[-1.0494156360524627, 5.764644726513884], 
                    "mean2":[6.9141859213193495, 5.59478222334447],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm.csv"})

test_cases.append({"name": "t_mix_2",
                    "mean1":[3.2782503568831554, 1.793573864746043], 
                    "mean2":[-0.9389365739797241, 2.05969562705752],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm.csv"})

test_cases.append({"name": "t_mix_3",
                    "mean1":[-1.1431749105000186, 2.0529438865919479], 
                    "mean2":[1.846096025789632, -0.5656381542446731],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm.csv"})

# Radius
test_cases.append({"name": "t_radius_1_5",
                    "mean1":[-1.1431749105000186, 2.0529438865919479], 
                    "mean2":[1.846096025789632, -0.5656381542446731],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm_radius_1.5.csv"})

test_cases.append({"name": "t_radius_2",
                    "mean1":[-1.1431749105000186, 2.0529438865919479], 
                    "mean2":[1.846096025789632, -0.5656381542446731],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm_radius_2.csv"})

test_cases.append({"name": "t_radius_2_5",
                    "mean1":[-1.1431749105000186, 2.0529438865919479], 
                    "mean2":[1.846096025789632, -0.5656381542446731],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm_radius_2.5.csv"})

test_cases.append({"name": "t_radius_3",
                    "mean1":[-1.1431749105000186, 2.0529438865919479], 
                    "mean2":[1.846096025789632, -0.5656381542446731],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False,
					"data":"gmm_radius_3.csv"})
'''

test_cases.append({"name": "t_multiple_1",
                    "mean1":[1.972979456933269, 0.7560544338372392], 
                    "mean2":[6.686047548070453, -0.1307013280230782],
					"mean3":[7.221786450205375, 6.262448613633105],
					"mean4":[-1.3977125108429398, 0.4590524145870889],
					"mean5":[3.307847427018068, 6.954851796948616],
                    "ld":1500,
                    "phis":[45, -45, 45, -45, 45],
                    "run":False,
					"data":"multiple_5.csv"})

test_cases.append({"name": "t_multiple_2",
                    "mean1":[0.072979456933269, -2.5560544338372392], 
                    "mean2":[6.886047548070453, -0.1307013280230782],
					"mean3":[9.221786450205375, 6.262448613633105],
					"mean4":[-5.3977125108429398, 2.0590524145870889],
					"mean5":[3.307847427018068, 6.954851796948616],
                    "ld":1500,
                    "phis":[45, -45, 45, -45, 45],
                    "run":False,
					"data":"multiple_5.csv"})

for test_case in test_cases:
	test_name = "{0}_{1}_{2}".format(test_case["name"], \
		int(test_case["phis"][0]) - int(test_case["phis"][1]), test_case["ld"])
	
	file_path = "jsons/{0}".format(test_name)

	if os.path.exists(file_path) == False:
		os.mkdir(file_path)
    
	with open("jsons/{0}/{0}.json".format(test_name), 'w') as outfile:
		json.dump(test_case, outfile)