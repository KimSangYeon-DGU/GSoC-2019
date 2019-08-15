import csv, json, os
import pandas as pd
import numpy as np
from param_utils import *

def gen_csv():
  df = pd.read_csv('data/faithful.csv', sep=',')
  dataset = df.to_numpy()
  dataset = np.transpose(dataset)

  tc_num = 1000
  x_std = np.std(dataset[0])
  y_std = np.std(dataset[1])
  gaussians = 2

  with open("testcase/tc.csv", "w") as csv_file:
    writer = csv.writer(csv_file, delimiter =",")

    for i in range(tc_num):
      means = get_initial_means(dataset, x_offset=x_std, 
                  y_offset=y_std, gaussians=gaussians)
      m1_x = means[0][0]
      m1_y = means[0][1]
      m2_x = means[1][0]
      m2_y = means[1][1]
      writer.writerow([i+1, m1_x, m1_y, m2_x, m2_y])

def cvt_to_json():
  phi_1 = 45
  phi_2 = 45
  ld = 1
  with open("testcase/tc.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    for i, row in enumerate(reader):
      data = {"name": "{0:05d}_comp".format(i+1),
                    "means":[[float(row[1]), float(row[2])],
                             [float(row[3]), float(row[4])]],
                    "covs":[[[0.08, 0.1], [0.1, 3.3]],
                            [[0.08, 0.1], [0.1, 3.3]]],
                    "alphas":[0.5, 0.5],
                    "phis":[phi_1, phi_2],
                    "gaussians":2,
                    "dimensionality":2,
                    "ld":ld,
                    "data":"faithful.csv"}

      test_name = "{0}_{1}_{2}".format(data["name"], phi_1 - phi_2, ld)
      
      file_path = "jsons/{0}".format(test_name)

      if os.path.exists(file_path) == False:
        os.mkdir(file_path)
        
      with open("jsons/{0}/{0}.json".format(test_name), 'w') as outfile:                
        json.dump(data, outfile)

if __name__ == "__main__":
  #gen_csv()
  cvt_to_json()