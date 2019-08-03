import numpy as np
from draw_utils import *
import csv, os

def record_graph(x_records,
                 nll_records,
                 constraint_records,
                 alpha_records,
                 phi_records,
                 gauss_records,
                 prob_records,
                 j_records,
                 test_name,
                 open_type='w'):

    graphs_path = "graphs/"+test_name
    if os.path.exists(graphs_path) == False:
        os.mkdir(graphs_path)

    # Set file names
    nll_file_name = '{0}_nll'.format(test_name)
    constraint_file_name = '{0}_constraint'.format(test_name)
    phi_file_name = '{0}_phi'.format(test_name)
    unnorm_gauss_file_name = '{0}_unnorm_gauss'.format(test_name)
    probs_file_name = '{0}_probs'.format(test_name)
    alpha_file_name = '{0}_alpha'.format(test_name)
    obj_file_name = '{0}_obj'.format(test_name)

    # Save data to csv format and a graph
    x_label=["Iteration"]

    # NLL
    nll_records = np.asarray(nll_records)
    ys_labels = ["NLL", "NLL"]

    draw_graph(x=x_records, ys=nll_records, x_label=x_label, ys_labels=ys_labels,
        file_name=nll_file_name+'.png', test_name=test_name)

    # Constraint
    constraint_records = np.asarray(constraint_records)
    ys_labels = ["Constraint", "Constraint"]

    draw_graph(x=x_records, ys=constraint_records, x_label=x_label, ys_labels=ys_labels, 
        file_name=constraint_file_name+'.png', test_name=test_name)

    # Phi
    phi_records = np.asarray(phi_records).transpose()
    ys_labels = ["Phi", "Phi_1", "Phi_2"]

    draw_graph(x=x_records, ys=phi_records, x_label=x_label, ys_labels=ys_labels, \
        file_name=phi_file_name+'.png', test_name=test_name)

    # Alpha
    alpha_records = np.asarray(alpha_records).transpose()
    ys_labels = ["Alphas", "Alpha_1", "Alpha_2"]

    draw_graph(x=x_records, ys=alpha_records, x_label=x_label, ys_labels=ys_labels,\
        file_name=alpha_file_name, test_name=test_name)

    # Unnormalized Gaussians
    gauss_records = np.asarray(gauss_records).transpose()
    ys_labels = ["Unnormalized Gaussians", "Unnormalized Gauss_1", "Unnormalized Gauss_2"]

    draw_graph(x=x_records, ys=gauss_records, x_label=x_label, \
        ys_labels=ys_labels, file_name=unnorm_gauss_file_name+'.png', \
        test_name=test_name)

    # Probability
    prob_records = np.asarray(prob_records).transpose()
    prob_records = np.vstack([prob_records, prob_records[0] + prob_records[1]])
    ys_labels = ["Sum of probability", "Sum of prob_1", "Sum of prob_2", "Sum of prob 1 and 2"]

    draw_graph(x=x_records, ys=prob_records, x_label=x_label, ys_labels=ys_labels, 
        file_name=probs_file_name+'.png', test_name=test_name)

    # Objective function
    j_records = np.asarray(j_records)
    ys_labels = ["Objective function", "Objective function"]

    draw_graph(x=x_records, ys=j_records, x_label=x_label, ys_labels=ys_labels, 
        file_name=obj_file_name+'.png', test_name=test_name)
