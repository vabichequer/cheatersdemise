# Imports and definition

import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
import collections as coll
import matplotlib as mpl
import networkx as nx
import seaborn as sns
import sklearn as skl
import pandas as pd
import sympy as sym
import numpy as np
import math
import json
import time
import sys
import ctd
import os
from sklearn.cluster import AgglomerativeClustering
from multiprocessing import Process, Manager
from sklearn.decomposition import PCA
from sklearn import preprocessing
from functools import partial
from ast import literal_eval

## Constants

#if sys.platform == 'win32':
#    from signal import signal, SIG_DFL
#else:
#    from signal import signal, SIGPIPE, SIG_DFL
#signal(SIGPIPE, SIG_DFL)

# define date format
date_format = '%Y-%m-%d %H:%M:%S'

# number of exercises
N_EXERCISES = 196

# Minimum number of exercises
MIN_EXERCISES = 10

# Time tolerance for exercises
try:
    tol = int(sys.argv[1])
except:
    sys.exit("Missing time tolerance argument. Exiting...")

trimming = tol #math.inf

cluster_tag = 'all'

## Definitions

# From the Main PIPE lib files:
#funcion que carga un fichero en un dataframe con el id del usuario en una columna y sus eventos en formato json en otra
def readDataFile(filename):
    # Leemos el fichero por lineas
    with open(filename) as f:
        users_events = f.readlines()
        
    # Parseamos a formato json cada linea
    listJson = list(map(json.loads, users_events));
   
    # Creamos el dataframe a partir de               
    df = pd.DataFrame.from_dict(listJson, orient='columns');
    return df

# sort functions
def take_second(elem):
    return elem[1]

def take_third(elem):
    return elem[2]

def take_size(elem):
    return len(elem)

# faster, but still slow
# fast subtractive convolutional algorithm? Nope. Not a convolution, as it would not need to get inverted
# Mathematical deduction on my scrapbook
def selectUsers(selected_users, tol, exercise_array_1, exercise_array_2, version, N_USERS, MIN_EXERCISES):
    for i in range(0, N_USERS - 1):
        for j in range(i + 1, N_USERS):
            time_dif = exercise_array_1[i] - exercise_array_2[j]
            if (version == 'XC'):
                time_dif = time_dif[time_dif <= 0]
            elif (version == 'CX'):
                time_dif = time_dif[time_dif >= 0]
            nbr_exercises = sum(abs(x) < tol for x in time_dif)
            if (nbr_exercises >= MIN_EXERCISES):
                print(version, 'added.', i, 'is user 1,', j, 'is user 2.', nbr_exercises, 'exercises.', tol, 'minutes tolerance')
                selected_users.append([i, j])

    print(version, 'thread finished.', version, 'added', len(selected_users), 'users.')
    return

def generate_pairs(data_array, user_id_data, scores_data, trimming, label, plot):
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    fig = plt.figure()
    
    user_pairs = []
    string_dump = []
    
    user_score_difference = []
    
    user_interaction_percentages = []
    
    example = ['(A)', '(B)', '(C)']

    example_iter = 0

    for i in range(0, len(data_array)):
        arrays = [[], [], [], []]           

        for j in range(0, 4):
            if(isinstance(data_array[i][1][j], list)):
                arrays[j] = calculateHistogram(data_array[i][1][j], trimming)
                user_1, user_2 = return_users(data_array[i][0])

        user_pairs.append((user_1, user_2))
            
        user1_id = int(user_id_data[user_1])
        user2_id = int(user_id_data[user_2])

        for p in range(0, len(scores_data)):
            if (user1_id == scores_data[p][0]):
                score1 = scores_data[p][1]

            if (user2_id == scores_data[p][0]):
                score2 = scores_data[p][1]

        user_score_difference.append(score1 - score2)
        
        if (plot):
            if ((user_1 == 115.0 and user_2 == 808.0) or (user_1 == 991.0 and user_2 == 3237.0) or (user_1 == 690.0 and user_2 == 1377.0)):
                n = len(fig.axes)
                for o in range(n):
                    fig.axes[o].change_geometry(n+1, 1, o+1)
                ax = fig.add_subplot(n+1, 1, n+1) 

                size = 25

                ax.set_title(example[example_iter] + str(user_1), fontsize=size)

                example_iter += 1
                
                string_dump.append(str('User ' + str(user_1) + ' (' + str(user1_id) +': ' + str(score1) + ') vs User ' +
                            str(user_2) + ' (' + str(user2_id) +': ' + str(score2) + ')' + ' || User bias: '
                            + str(data_array[i][2])))
                
                ax.set_xlabel('dt (User 1, User 2)', fontsize=size)
                ax.set_ylabel('Number of occurences', fontsize=size)
                ax.tick_params(axis='both', which='both', labelsize=size)

                new_xc = [i for i in arrays[1] if i < 0] + [i for i in arrays[2] if i >= 0]
                new_cx = [i for i in arrays[1] if i >= 0] + [i for i in arrays[2] if i < 0]

                ax.hist([arrays[0], new_xc, new_cx, arrays[3]], color=['blue', 'orange', 'green', 'red'],
                        label=label, bins=100, stacked=True)         
                ax.legend(prop={'size': size})

        new_xc = [i for i in arrays[1] if i < 0] + [i for i in arrays[2] if i >= 0]
        new_cx = [i for i in arrays[1] if i >= 0] + [i for i in arrays[2] if i < 0]

        total_ex = len(arrays[0]) + len(arrays[1]) + len(arrays[2]) + len(arrays[3])
                
        user_interaction_percentages.append((len(arrays[0])/total_ex, (len(new_xc))/total_ex, len(arrays[3])/total_ex))
            
    return user_pairs, user_score_difference, fig, string_dump, user_interaction_percentages

def get_number_of_exercises(data, tol):
    #nbr_exercises_over = sum(((x >= 0) and (x < tol)) for x in data)
    #nbr_exercises_under = sum(((x <= 0) and (x > -tol)) for x in data)

    nbr_exercises_over = 0
    nbr_exercises_under = 0
    
    nbr_exercises = sum((not pd.isnull(x) and abs(x) < tol) for x in data)
    
    for i in range(0, len(data)):
        x = data[i]
        
        if ((x >= 0) and (x < tol)):
            nbr_exercises_over = nbr_exercises_over + 1
        elif ((x <= 0) and (x > -tol)):
            nbr_exercises_under = nbr_exercises_under - 1
    
    return nbr_exercises, nbr_exercises_over, nbr_exercises_under

def return_users(pair):
    user_1 = pair[0]
    user_2 = pair[1]
    
    return user_1, user_2

def type_separation(all_user_pairs, all_time_differences, tol):
    type_array = []
    type_1 = []
    type_2 = []
    
    # This part goes over the 3 arrays inside all_user_pairs and joins the CC, XC and XX
    # cases by searching through every pair and checking if there are matches between
    # the arrays. Every iteration of 'outer_loop' will exclude a case, for example:
    # The array is ordered by size. Therefore, if the order happens to be CC, then XC then XX,
    # when the CC search is finished, it will jump for the XC and search for XX matches.
    # When the search for XC is finished, it will add the cases where XX is alone.
    
    for outer_loop in range(0, 4):   
        ARRAY_TIER = outer_loop             
        for i in range(0, len(all_user_pairs[ARRAY_TIER])):  
            ARRAY_TIER = outer_loop
            
            pair = [np.nan, np.nan, np.nan, np.nan]
            td = [np.nan, np.nan, np.nan, np.nan]
            nbr_ex = [np.nan, np.nan, np.nan, np.nan]
            nbr_over = [np.nan, np.nan, np.nan, np.nan]
            nbr_under = [np.nan, np.nan, np.nan, np.nan]
            include_check = [np.nan, np.nan, np.nan, np.nan]
            time_differences = [np.nan, np.nan, np.nan, np.nan]
            users = np.zeros((4, 2))
                       
            users[ARRAY_TIER][0], users[ARRAY_TIER][1] = return_users(all_user_pairs[ARRAY_TIER][i])
            nbr_ex[ARRAY_TIER], nbr_over[ARRAY_TIER], nbr_under[ARRAY_TIER] = get_number_of_exercises(all_time_differences[ARRAY_TIER][i], tol)
            pair[ARRAY_TIER] = all_user_pairs[ARRAY_TIER][i]
            td[ARRAY_TIER] = all_time_differences[ARRAY_TIER][i]

            ARRAY_TIER = outer_loop + 1
            if (ARRAY_TIER < 4):
                for j in range(0, len(all_user_pairs[ARRAY_TIER])):
                    users[ARRAY_TIER][0], users[ARRAY_TIER][1] = return_users(all_user_pairs[ARRAY_TIER][j])
                    if ((users[outer_loop][0] == users[ARRAY_TIER][0]) and (users[outer_loop][1] == users[ARRAY_TIER][1])):
                        nbr_ex[ARRAY_TIER], nbr_over[ARRAY_TIER], nbr_under[ARRAY_TIER] = get_number_of_exercises(all_time_differences[ARRAY_TIER][j], tol)
                        pair[ARRAY_TIER] = all_user_pairs[ARRAY_TIER][j]
                        td[ARRAY_TIER] = all_time_differences[ARRAY_TIER][j]
                        all_user_pairs[ARRAY_TIER].pop(j)
                        all_time_differences[ARRAY_TIER].pop(j)
                        break
            
            ARRAY_TIER = outer_loop + 2
            if (ARRAY_TIER < 4):
                for k in range(0, len(all_user_pairs[ARRAY_TIER])):
                    users[ARRAY_TIER][0], users[ARRAY_TIER][1] = return_users(all_user_pairs[ARRAY_TIER][k])
                    if ((users[outer_loop][0] == users[ARRAY_TIER][0]) and (users[outer_loop][1] == users[ARRAY_TIER][1])):
                        nbr_ex[ARRAY_TIER], nbr_over[ARRAY_TIER], nbr_under[ARRAY_TIER] = get_number_of_exercises(all_time_differences[ARRAY_TIER][k], tol)
                        pair[ARRAY_TIER] = all_user_pairs[ARRAY_TIER][k]
                        td[ARRAY_TIER] = all_time_differences[ARRAY_TIER][k]
                        all_user_pairs[ARRAY_TIER].pop(k) 
                        all_time_differences[ARRAY_TIER].pop(k)   
                        break
                        
            ARRAY_TIER = outer_loop + 3
            if (ARRAY_TIER < 4):
                for l in range(0, len(all_user_pairs[ARRAY_TIER])):
                    users[ARRAY_TIER][0], users[ARRAY_TIER][1] = return_users(all_user_pairs[ARRAY_TIER][l])
                    if ((users[outer_loop][0] == users[ARRAY_TIER][0]) and (users[outer_loop][1] == users[ARRAY_TIER][1])):
                        nbr_ex[ARRAY_TIER], nbr_over[ARRAY_TIER], nbr_under[ARRAY_TIER] = get_number_of_exercises(all_time_differences[ARRAY_TIER][l], tol)
                        pair[ARRAY_TIER] = all_user_pairs[ARRAY_TIER][l]
                        td[ARRAY_TIER] = all_time_differences[ARRAY_TIER][l]
                        all_user_pairs[ARRAY_TIER].pop(l) 
                        all_time_differences[ARRAY_TIER].pop(l)   
                        break

            total_ex = np.nansum(nbr_ex)
            total_over = np.nansum(nbr_over)
            total_under = np.nansum(nbr_under)

            user_bias = (total_over + total_under) / total_ex
            
            type_array.append((users[outer_loop], td, user_bias, total_ex))          

# DEBUG
#            print(outer_loop, ':')
#            print('\t User:', users[outer_loop][0], 'vs User:', users[outer_loop][1])
#            print('\t Nbr exercises:', total_ex, 'Over:', total_over, 'Under:', total_under)
#            print('\t User bias:', user_bias)
    
    #type_array.sort(key=take_third)
    
    return type_array

def computeMetrics(tol, time_difference):
    avg_dist = np.empty((len(time_difference), len(time_difference)))
    avg_dist.fill(np.nan)
    std_dev = np.empty((len(time_difference), len(time_difference)))
    std_dev.fill(np.nan)
    
    for k in range(0, len(time_difference)):
        nbr_exercises = np.zeros(len(time_difference))
        for j in range(0, len(time_difference)):            
            temp_std_dev = np.empty(N_EXERCISES)
            temp_std_dev.fill(np.nan)
            temp_avg = 0
            for i in range(0, N_EXERCISES):                
                temp_std_dev[i] = abs(time_difference[k][i])
                if (abs(time_difference[k][i]) < tol):
                    nbr_exercises[j] = nbr_exercises[j] + 1
                    temp_avg = temp_avg + time_difference[k][i]
            avg_dist[k][j] = temp_avg/nbr_exercises[j]
            std_dev[k][j] = np.nanstd(temp_std_dev)
                
    return avg_dist, std_dev

def plotHeatmap(data, title):
    plt.figure(figsize=(50, 50))
    ax = plt.axes()
    sns.set(font_scale=3)
    mask = np.zeros_like(data, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    ax.set_title(title)
    sns.heatmap(data, mask=mask, square=True, linewidths=3, cbar_kws={"shrink": .5}, ax = ax)

def calculateHistogram(time_differences, trimming, minutes=True):    
    temp_hist = np.empty(N_EXERCISES)
    temp_hist.fill(np.nan)

    for i in range(0, N_EXERCISES):
        if(abs(time_differences[i]) < trimming):
            temp_hist[i] = time_differences[i]
            if not minutes:
                temp_hist[i] *= 60

    temp_hist = temp_hist[~np.isnan(temp_hist)]    
    return temp_hist

def distance(x1, y1):
    x = np.linspace(-1, 1, 201)
    distance = []
    
    return min(((x - x1)**2 + (x**9 - y1)**2)**(1/2))

### Check IP addresses
# This methos checks the ips from each pair in 'users' and adds them to a list, excluding repetitons. 
# It also checks how many unique exercises were done with an IP for each in the pair.

def check_ip_addresses(users, df):
    df_ip = readDataFile("eventos_conIp.json")
    ip_addrs = []
    
    for i in range(0, len(users)):
        temp_1 = []
        temp_2 = []
        count_1 = []
        count_2 = []
        unique_ex_1 = []
        unique_ex_2 = []
        
        for j in range(1, len(df_ip.Usuario)):
            if (int(df_ip.Usuario[j]) == int(df.Usuario[users[i][2]])):
                for k in range(0, len(df_ip.Eventos[j])):
                    if(df_ip.Eventos[j][k]['evento'] == 'problem_check'):
                        if (len(temp_1) > 0):
                            if (df_ip.Eventos[j][k]['ip'] in temp_1):
                                if (not (df_ip.Eventos[j][k]['id_problema']) in unique_ex_1):
                                    index = temp_1.index(df_ip.Eventos[j][k]['ip'])
                                    count_1[index] = count_1[index] + 1
                                    unique_ex_1.append(df_ip.Eventos[j][k]['id_problema'])
                            else:
                                temp_1.append(df_ip.Eventos[j][k]['ip'])
                                count_1.append(1)
                                unique_ex_1.append(df_ip.Eventos[j][k]['id_problema'])
                        else:
                            temp_1.append(df_ip.Eventos[j][k]['ip'])
                            count_1.append(1)
                            unique_ex_1.append(df_ip.Eventos[j][k]['id_problema'])
                            
            if (int(df_ip.Usuario[j]) == int(df.Usuario[users[i][3]])):
                for k in range(0, len(df_ip.Eventos[j])):
                    if(df_ip.Eventos[j][k]['evento'] == 'problem_check'):
                        if (len(temp_2) > 0):
                            if (df_ip.Eventos[j][k]['ip'] in temp_2):
                                if (not (df_ip.Eventos[j][k]['id_problema']) in unique_ex_2):
                                    index = temp_2.index(df_ip.Eventos[j][k]['ip'])
                                    count_2[index] = count_2[index] + 1
                                    unique_ex_2.append(df_ip.Eventos[j][k]['id_problema'])
                            else:
                                temp_2.append(df_ip.Eventos[j][k]['ip'])
                                count_2.append(1)
                                unique_ex_2.append(df_ip.Eventos[j][k]['id_problema'])
                        else:
                            temp_2.append(df_ip.Eventos[j][k]['ip'])
                            count_2.append(1)
                            unique_ex_2.append(df_ip.Eventos[j][k]['id_problema'])
     
        ip_addrs.append([temp_1, temp_2, count_1, count_2])

    return ip_addrs

def check_material_usage(users, df):
    material_usage_rate = np.zeros((len(users), 2))
    
    for i in range(0, len(users)):
        analysed_exercises_1 = []
        analysed_exercises_2 = []
        #print('-'*100)
        for j in range(0, len(df.Usuario)):
            total_1 = 0
            total_2 = 0
            material_usage_1 = 0
            material_usage_2 = 0

            if (int(df.Usuario[j]) == int(df.Usuario[users[i][2]])):
                for k in range(0, len(df.Eventos[j])):
                    if(df.Eventos[j][k]['evento'] == 'problem_check'):
                        if (not(df.Eventos[j][k]['id_problema'] in analysed_exercises_1)):
                            total_1 = total_1 + 1
                            analysed_exercises_1.append(df.Eventos[j][k]['id_problema'])
                    else:
                        if (df.Eventos[j][k]['evento'] != 'error_json'):
                            material_usage_1 = material_usage_1 + 1
                            total_1 = total_1 + 1
                
                material_usage_rate[i][0] = (material_usage_1/total_1)
                #print('User 1 (', users[i][2], ') material usage rate: ', material_usage_1/total_1)

            if (int(df.Usuario[j]) == int(df.Usuario[users[i][3]])):
                for k in range(0, len(df.Eventos[j])):
                    if(df.Eventos[j][k]['evento'] == 'problem_check'):
                        if (not(df.Eventos[j][k]['id_problema'] in analysed_exercises_2)):
                            total_2 = total_2 + 1
                            analysed_exercises_2.append(df.Eventos[j][k]['id_problema'])
                    else:
                        if (df.Eventos[j][k]['evento'] != 'error_json'):
                            material_usage_2 = material_usage_2 + 1
                            total_2 = total_2 + 1

                material_usage_rate[i][1] = (material_usage_2/total_2)
                #print('User 2 (', users[i][3], ') material usage rate: ', material_usage_2/total_2)
    
    return material_usage_rate

if (trimming < tol):
    raise ValueError('The trimming value is lower than the tolerance. The code will not work like this.')
else:
    print('Libraries and definitions loaded correctly.')
