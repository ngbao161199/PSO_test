import source_code as ps
from source_code.utils.functions import opt_function as fx
from source_code.backend.topology import Star, Ring
from source_code.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd 
import sys
import os

def input(choice, function, para, indi, mseed):
    if choice == 1:
        folder_path = "logs/" + str(para) + "_" + str(indi) + "_" + "ring" 
        plot_folder = "plots/" + str(para) + "_" + str(indi) + "_" + "ring"
        x= Ring()
    if choice == 2:
        folder_path = "logs/" + str(para) + "_" + str(indi) + "_" + "star"
        plot_folder = "plots/" + str(para) + "_" + str(indi) + "_" + "star" 
        x= Star()
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    if para == 2:
        if function == 1:
            output_path = folder_path + "/Log" + "_" + str(mseed) + "_ras" + ".log"
            plot_path = plot_folder + "/plot" + "_" + str(mseed) + "_ras" + ".png"
            f = fx.rastrigin
        if function == 2:
            output_path = folder_path + "/Log" + "_" + str(mseed) + "_beal" + ".log"
            plot_path = plot_folder + "/plot" + "_" + str(mseed) + "_beal" + ".png"  
            f= fx.beale
        if function == 3:
            output_path = folder_path + "/Log" + "_" + str(mseed) + "_himme" + ".log"
            plot_path = plot_folder + "/plot" + "_" + str(mseed) + "_himme" + ".png"  
            f= fx.himmelblau
        if function == 4:
            output_path = folder_path + "/Log" + "_" + str(mseed) + "_cross" + ".log"
            plot_path = plot_folder + "/plot" + "_" + str(mseed) + "_cross" + ".png"  
            f= fx.crossintray
        return output_path, plot_path, f , x
    
    if para == 10:
        output_path = folder_path + "/Log" + "_" + str(mseed) + "_ras" + ".log"
        plot_path = plot_folder + "/plot" + "_" + str(mseed) + "_ras" + ".png"
        f = fx.rastrigin
        return output_path, plot_path, f , x

argv = sys.argv[0:]

mseed = argv[1]

mseed = (int)(mseed)

# Set-up hyperparameters
options = {'c1': 1.49618, 'c2': 1.49618, 'w':0.7298, 'k':5, 'p': 2}
# Call instance of PSO

opt = [1, 2, 3, 4] # 'Rastrigin', 'Beale', 'Himmelblau', 'Cross_in_tray'
neighbor = [1,2] # 1 - star, 2 -  ring

my_columns = ['name', 'c1=c2', 'w', 'k', 'p', 'indi', 'para', 'function', 'neighbor', 'best_cost', 'best_pos']
if not os.path.isfile('summary.csv'):
    my_columns = np.array(my_columns)
    my_columns = my_columns.reshape(-1, my_columns.shape[0])

    df = pd.DataFrame(my_columns)
    df.to_csv("summary.csv", mode='a', header=None)

if len(argv) != 2:
    print('Wrong!!!\nPlease use format: python script seed')
    print('Example: python GA.py 17520273')
else:
    go = 0
    para = 2
    indi = 32
    np.random.seed(mseed)
    random.seed(mseed)

    # case for para = 2 
    for choice in neighbor:
        for function in opt:
            output_path, plot_path, f , x = input(choice, function, para, indi, mseed)
            
            #####################
                # create dataframe
            ######################
            list_attr = [output_path, options['c1'], options['w'], options['k'], options['p'], indi, para]
            
            # add function
            if function == 1:
                list_attr.append('Rastrigin')
            elif function == 2:
                list_attr.append('Beale')
            elif function == 3:
                list_attr.append('Himmelblau')
            else:
                list_attr.append('Cross_in_tray')
            
            # add neighbor
            if choice == 1:
                list_attr.append('star')
            else:
                list_attr.append('ring')
            
            optimizer = ps.pso.General(log_path = output_path, n_particles=indi, dimensions=para, options=options, topology = x)
            best_cost, best_pos = optimizer.optimize(f, log_path=output_path, iters=50, maxiter=100, para=para)

            # add best_cost, best_pos to data frame
            list_attr.append(best_cost)
            list_attr.append(best_pos)
            arr_attr = np.array(list_attr)
            arr_attr = arr_attr.reshape(-1, arr_attr.shape[0])

            df = pd.DataFrame(arr_attr)
            df.to_csv("summary.csv", mode='a', header=None)

            optimizer.cost_history
            plt.figure()
            plot_cost_history(optimizer.cost_history)
            plt.savefig(plot_path)

            #PSO(seed, function, SearchDomain, indi, para, gen, choice, output_path)

    # case para = 10
    para = 10
    indi = [128, 256, 512, 1024, 2048]

    while(go<10):
        for num_individuals in indi:
            for choice in neighbor:
                for function in opt:
                    output_path, plot_path, f , x = input(choice, function, para, num_individuals, mseed)

                    #####################
                        # create dataframe
                    ######################
                    list_attr = [output_path, options['c1'], options['w'], options['k'], options['p'], num_individuals, para]
                    
                    # add function
                    if function == 1:
                        list_attr.append('Rastrigin')
                    elif function == 2:
                        list_attr.append('Beale')
                    elif function == 3:
                        list_attr.append('Himmelblau')
                    else:
                        list_attr.append('Cross_in_tray')
                    
                    # add neighbor
                    if choice == 1:
                        list_attr.append('star')
                    else:
                        list_attr.append('ring')

                    optimizer = ps.pso.General(log_path = output_path, n_particles=num_individuals, dimensions=para, options=options, topology = x)
                    best_cost, best_pos = optimizer.optimize(f, log_path=output_path, iters=50, maxiter=100, para=para)

                    # add best_cost, best_pos to data frame
                    list_attr.append(best_cost)
                    list_attr.append(best_pos)
                    arr_attr = np.array(list_attr)
                    arr_attr = arr_attr.reshape(-1, arr_attr.shape[0])

                    df = pd.DataFrame(arr_attr)
                    df.to_csv("summary.csv", mode='a', header=None)

                    optimizer.cost_history
                    plt.figure()
                    plot_cost_history(optimizer.cost_history)
                    plt.savefig(plot_path)
                    #PSO10(seed, function, SearchDomain, num_individuals, para, gen, choice, output_path)
        go += 1 
        mseed += 1
