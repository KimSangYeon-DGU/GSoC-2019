
test_cases = []

## Initialization
# Test case 1
test_cases.append({"name": "t_init_1",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":1500,
                    "phis":[0, 0],
                    "run":False})

test_cases.append({"name": "t_init_1",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_init_1",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})

# Test case 2
test_cases.append({"name": "t_init_2",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":1500,
                    "phis":[0, 0],
                    "run":False})

test_cases.append({"name": "t_init_2",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_init_2",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})

# Test case 3
test_cases.append({"name": "t_init_3",
                    "mean1":[4.616385494792178, 68.97139287485163], 
                    "mean2":[4.73416217991247, 70.48443049223583],
                    "ld":1500,
                    "phis":[0, 0],
                    "run":False})

test_cases.append({"name": "t_init_3",
                    "mean1":[4.616385494792178, 68.97139287485163], 
                    "mean2":[4.73416217991247, 70.48443049223583],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_init_3",
                    "mean1":[4.616385494792178, 68.97139287485163], 
                    "mean2":[4.73416217991247, 70.48443049223583],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})

# Test case 4
test_cases.append({"name": "t_init_4",
                    "mean1":[3.5335808453329793, 60.79723193882826], 
                    "mean2":[3.748786959785587, 46.017018024467745],
                    "ld":1500,
                    "phis":[0, 0],
                    "run":False})

test_cases.append({"name": "t_init_4",
                    "mean1":[3.5335808453329793, 60.79723193882826], 
                    "mean2":[3.748786959785587, 46.017018024467745],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_init_4",
                    "mean1":[3.5335808453329793, 60.79723193882826], 
                    "mean2":[3.748786959785587, 46.017018024467745],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})

# Test case 5
test_cases.append({"name": "t_init_5",
                    "mean1":[4.399318766072071, 63.982790484402784], 
                    "mean2":[2.511548424664534, 90.2446329311453],
                    "ld":1500,
                    "phis":[0, 0],
                    "run":False})

test_cases.append({"name": "t_init_5",
                    "mean1":[4.399318766072071, 63.982790484402784], 
                    "mean2":[2.511548424664534, 90.2446329311453],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_init_5",
                    "mean1":[4.399318766072071, 63.982790484402784], 
                    "mean2":[2.511548424664534, 90.2446329311453],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})

## Distance
# This test case is the same with t_init_1 with phi 180
test_cases.append({"name": "t_dist_1",
                    "mean1":[2.756031811312966, 76.62447648112042],
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})


test_cases.append({"name": "t_dist_2",
                    "mean1":[2.3, 72.6],
                    "mean2":[3.3, 92.4],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":True})

test_cases.append({"name": "t_dist_3",
                    "mean1":[1.5, 68.6],
                    "mean2":[3.3, 92.4],
                    "ld":1500,
                    "phis":[90, -90],
                    "run":False})

## Lambda selection
# Test case 1
test_cases.append({"name": "t_lambda_1",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_1",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_1",
                    "mean1":[2.756031811312966, 76.62447648112042], 
                    "mean2":[2.9226572802266397, 88.3509418943818],
                    "ld":5000,
                    "phis":[45, -45],
                    "run":False})

# Test case 2
test_cases.append({"name": "t_lambda_2",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_2",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_2",
                    "mean1":[4.171021823127277, 83.66322004888708], 
                    "mean2":[1.781079954983019, 95.411542531776],
                    "ld":5000,
                    "phis":[45, -45],
                    "run":False})

# Test case 3
test_cases.append({"name": "t_lambda_3",
                    "mean1":[4.616385494792178, 68.97139287485163], 
                    "mean2":[4.73416217991247, 70.48443049223583],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_3",
                    "mean1":[4.616385494792178, 68.97139287485163], 
                    "mean2":[4.73416217991247, 70.48443049223583],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_3",
                    "mean1":[4.616385494792178, 68.97139287485163], 
                    "mean2":[4.73416217991247, 70.48443049223583],
                    "ld":5000,
                    "phis":[45, -45],
                    "run":False})

# Test case 4
test_cases.append({"name": "t_lambda_4",
                    "mean1":[3.5335808453329793, 60.79723193882826], 
                    "mean2":[3.748786959785587, 46.017018024467745],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_4",
                    "mean1":[3.5335808453329793, 60.79723193882826], 
                    "mean2":[3.748786959785587, 46.017018024467745],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_4",
                    "mean1":[3.5335808453329793, 60.79723193882826], 
                    "mean2":[3.748786959785587, 46.017018024467745],
                    "ld":5000,
                    "phis":[45, -45],
                    "run":False})

# Test case 5
test_cases.append({"name": "t_lambda_5",
                    "mean1":[4.399318766072071, 63.982790484402784], 
                    "mean2":[2.511548424664534, 90.2446329311453],
                    "ld":100,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_5",
                    "mean1":[4.399318766072071, 63.982790484402784], 
                    "mean2":[2.511548424664534, 90.2446329311453],
                    "ld":1500,
                    "phis":[45, -45],
                    "run":False})

test_cases.append({"name": "t_lambda_5",
                    "mean1":[4.399318766072071, 63.982790484402784], 
                    "mean2":[2.511548424664534, 90.2446329311453],
                    "ld":5000,
                    "phis":[45, -45],
                    "run":False})