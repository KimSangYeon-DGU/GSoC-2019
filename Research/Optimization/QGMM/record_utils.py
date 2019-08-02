import numpy as np
from draw_utils import *
import csv

def record_csv_graph(x_records,
                     nll_records,
                     constraint_records,
                     alpha_records,
                     phi_records,
                     gauss_records,
                     prob_records,
                     j_records,
                     mean1_records,
                     mean2_records,
                     cov1_records,
                     cov2_records,
                     test_name,
                     open_type='w'):

    # Set file names
    nll_file_name = '{0}_nll'.format(test_name)
    constraint_file_name = '{0}_constraint'.format(test_name)
    phi_file_name = '{0}_phi'.format(test_name)
    unnorm_gauss_file_name = '{0}_unnorm_gauss'.format(test_name)
    probs_file_name = '{0}_probs'.format(test_name)
    alpha_file_name = '{0}_alpha'.format(test_name)
    obj_file_name = '{0}_obj'.format(test_name)
    mean1_file_name = '{0}_mean1'.format(test_name)
    mean2_file_name = '{0}_mean2'.format(test_name)
    cov1_file_name = '{0}_cov1'.format(test_name)
    cov2_file_name = '{0}_cov2'.format(test_name)
    csv_path = './csvs/{0}'.format(test_name)


    # Save data to csv format and a graph
    x_label=["Iteration"]

    # NLL
    nll_records = np.asarray(nll_records)
    ys_labels = ["NLL", "NLL"]
    
    with open(csv_path+'/'+nll_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, nll_records))

    draw_graph(x=x_records, ys=nll_records, x_label=x_label, ys_labels=ys_labels,
        file_name=nll_file_name+'.png', test_name=test_name)

    # Constraint
    constraint_records = np.asarray(constraint_records)
    ys_labels = ["Constraint", "Constraint"]

    with open(csv_path+'/'+constraint_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, constraint_records))

    draw_graph(x=x_records, ys=constraint_records, x_label=x_label, ys_labels=ys_labels, 
        file_name=constraint_file_name+'.png', test_name=test_name)

    # Phi
    phi_records = np.asarray(phi_records).transpose()
    ys_labels = ["Phi", "Phi_1", "Phi_2"]

    with open(csv_path+'/'+phi_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, phi_records[0], phi_records[1]))

    draw_graph(x=x_records, ys=phi_records, x_label=x_label, ys_labels=ys_labels, \
        file_name=phi_file_name+'.png', test_name=test_name)

    # Alpha
    alpha_records = np.asarray(alpha_records).transpose()
    ys_labels = ["Alphas", "Alpha_1", "Alpha_2"]

    with open(csv_path+'/'+alpha_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, alpha_records[0], alpha_records[1]))

    draw_graph(x=x_records, ys=alpha_records, x_label=x_label, ys_labels=ys_labels,\
        file_name=alpha_file_name, test_name=test_name)

    # Unnormalized Gaussians
    gauss_records = np.asarray(gauss_records).transpose()
    ys_labels = ["Unnormalized Gaussians", "Unnormalized Gauss_1", "Unnormalized Gauss_2"]

    with open(csv_path+'/'+unnorm_gauss_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, gauss_records[0], gauss_records[1]))

    draw_graph(x=x_records, ys=gauss_records, x_label=x_label, \
        ys_labels=ys_labels, file_name=unnorm_gauss_file_name+'.png', \
        test_name=test_name)

    # Probability
    prob_records = np.asarray(prob_records).transpose()
    prob_records = np.vstack([prob_records, prob_records[0] + prob_records[1]])
    ys_labels = ["Sum of probability", "Sum of prob_1", "Sum of prob_2", "Sum of prob 1 and 2"]

    with open(csv_path+'/'+probs_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, prob_records[0], prob_records[1], prob_records[2]))

    draw_graph(x=x_records, ys=prob_records, x_label=x_label, ys_labels=ys_labels, 
        file_name=probs_file_name+'.png', test_name=test_name)

    # Objective function
    j_records = np.asarray(j_records)
    ys_labels = ["Objective function", "Objective function"]

    with open(csv_path+'/'+obj_file_name, open_type) as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, j_records))

    draw_graph(x=x_records, ys=j_records, x_label=x_label, ys_labels=ys_labels, 
        file_name=obj_file_name+'.png', test_name=test_name)


    # Means
    mean1_records = np.asarray(mean1_records).transpose()
    
    with open(csv_path+'/'+mean1_file_name, 'a+') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, mean1_records[0], mean1_records[1]))

    mean2_records = np.asarray(mean2_records).transpose()
    
    with open(csv_path+'/'+mean2_file_name, 'a+') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, mean2_records[0], mean2_records[1]))

    # Covs
    cov1_records = np.asarray(cov1_records).transpose()

    with open(csv_path+'/'+cov1_file_name, 'a+') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, cov1_records[0], cov1_records[1],
                                    cov1_records[2], cov1_records[3]))

    cov2_records = np.asarray(cov2_records).transpose()

    with open(csv_path+'/'+cov2_file_name, 'a+') as myfile:
        wr = csv.writer(myfile, delimiter=',')
        wr.writerows(zip(x_records, cov2_records[0], cov2_records[1],
                                    cov2_records[2], cov2_records[3]))