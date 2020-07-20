#!/usr/bin/env python
# coding: utf-8

# # Cheaters demise code

# The previous code defined the clusters of suspicious users, which were characterized as those who didn't watch (almost) any videos and documents and got all the answers right. These are thought to be users who deal with fake and real accounts, counting errors on the fakes and taking the right answers on the real ones.
# 
# Since there is a defined cluster of suspicious users, the following code will try to identify the activity of these users and see if they can be considered as cheaters or not, depending on timing and improvement. By taking into account that timing would be equal to the time difference between an answer from one user and the other, there may be a correlation between them, since after finding the real answer on a fake account, the user would input it into his real account in a short period. Also, improvement is a metric that can differentiate between a user that coincidently did it right after another and a cheater, by measuring if the user got the wrong answer or not. If it did get the wrong answer, than he's not using a fake account.
# 
# One course of action is to create a graph of exercise (i) by user(j) and the time it took the user to finish it.

#  # Import all definitions and libraries

# In[402]:

import importlib
import Definitions
import ctd

importlib.reload(Definitions)
importlib.reload(ctd)

from Definitions import *

print("Code started. Time window of: ", tol)

# # Read scores and user data

if (NEW_DATASET):
    df = readDataFile("eventos_conIp_Ed2.json")
    scores_csv = pd.read_csv("UAMx_Android301x_3T2015_grade_report_2015-12-10-1243_sanitized.csv")
    dataset_prefix = ''
else:
    df = readDataFile("eventos_final.json")
    scores_csv = pd.read_csv("UAMx_Android301x_1T2015_grade_report_2015-04-21-1145_sanitized.csv")
    dataset_prefix = 'original_data/'


N_USERS = len(df.Usuario)

# create the exercise list
correctExercises = np.empty((N_USERS, N_EXERCISES))
correctExercises.fill(np.nan)

correctExercisesCount = np.empty(N_USERS)
correctExercisesCount.fill(0)

wrongExercises = np.empty((N_USERS, N_EXERCISES))
wrongExercises.fill(np.nan)

wrongExercisesCount = np.empty(N_USERS)
wrongExercisesCount.fill(0)

scores = []

print_verb("Logging scores...")

for i in range(1, len(df.Usuario)):
    try:
        if (df.Usuario[i] != 'None'):
            score = scores_csv.loc[scores_csv['id'] == int(df.Usuario[i])]['grade'].item()
            scores.append((int(df.Usuario[i]), score))
        else:
            scores.append((np.nan, np.nan))
    except:                      
        scores.append((int(df.Usuario[i]), np.nan))

print_verb("Finished logging scores.")
print_verb("Logging exercises...")

exercise_dump_exists = os.path.isfile(dataset_prefix + 'correct_exercises.npy')

if (exercise_dump_exists and not REWRITE_DUMPS):
    correctExercises = np.load(dataset_prefix + 'correct_exercises.npy', allow_pickle=True)
    correctExercisesCount = np.load(dataset_prefix + 'correct_exercises_count.npy', allow_pickle=True)
    wrongExercises = np.load(dataset_prefix + 'wrong_exercises.npy', allow_pickle=True)
    wrongExercisesCount = np.load(dataset_prefix + 'wrong_exercises_count.npy', allow_pickle=True)
else:
    # create the exercise list
    correctExercises = np.empty((N_USERS, N_EXERCISES))
    correctExercises.fill(np.nan)

    correctExercisesCount = np.empty(N_USERS)
    correctExercisesCount.fill(0)

    wrongExercises = np.empty((N_USERS, N_EXERCISES))
    wrongExercises.fill(np.nan)

    wrongExercisesCount = np.empty(N_USERS)
    wrongExercisesCount.fill(0)

    if (TIME_LIMIT):
        time_lim = 1428105600000 #1 month, from march to april
    else:
        time_lim = sys.maxint

    for i in range(0, len(df.Usuario)):
        for j in range(0, len(df.Eventos[i])):
            if (df.Eventos[i][j]['evento'] == 'problem_check'):
                #print_verb('\t problem_check')
                if (df.Eventos[i][j]['resultados'] != []):
                    #print_verb('\t \t results')
                    # convert date time to epoch time for a better comparison
                    time_split = df.Eventos[i][j]['tiempo'].split('T')
                    time_tuple = time.strptime(time_split[0] + ' ' + time_split[1][:8], date_format)
                    time_epoch = time.mktime(time_tuple)   
                    #print_verb(time_epoch)
                    if (time_epoch < time_lim):
                        if(df.Eventos[i][j]['resultados'][0]['correcto'] == 'True'):
                            #print_verb('\t \t \t right')
                            correctExercises[i][int(df.Eventos[i][j]['id_problema']) - 1] = time_epoch
                            correctExercisesCount[i] = correctExercisesCount[i] + 1
                        elif(df.Eventos[i][j]['resultados'][0]['correcto'] == 'False'):
                            #print_verb('\t \t \t wrong')
                            wrongExercises[i][int(df.Eventos[i][j]['id_problema']) - 1] = time_epoch
                            num_intentos = int(df.Eventos[i][j]['num_intentos'])
                            wrongExercisesCount[i] = (wrongExercisesCount[i] - num_intentos + 1) + num_intentos
        #clear()
        #print_verb('Exercises:', math.ceil(i*100/N_USERS), '% done')

    np.save(dataset_prefix + 'correct_exercises.npy', correctExercises, allow_pickle=True)
    np.save(dataset_prefix + 'correct_exercises_count.npy', correctExercisesCount, allow_pickle=True)
    np.save(dataset_prefix + 'wrong_exercises.npy', wrongExercises, allow_pickle=True)
    np.save(dataset_prefix + 'wrong_exercises_count.npy', wrongExercisesCount, allow_pickle=True)

print_verb('Ended logging.')

N_USERS = len(df)


# ## Cutting down the size of the arrays

# In[386]:


# Cutting down the size of the arrays

print_verb(df.shape, correctExercises.shape, wrongExercises.shape)

users_to_delete = []

for i in range(0, N_USERS):
    nbr_attempts = wrongExercisesCount[i] + correctExercisesCount[i]
    if (nbr_attempts < MIN_EXERCISES):
        df = df.drop([i])
        users_to_delete.append(i)

correctExercises = np.delete(correctExercises, users_to_delete, 0)
wrongExercises = np.delete(wrongExercises, users_to_delete, 0)
correctExercisesCount = np.delete(correctExercisesCount, users_to_delete, 0)
wrongExercisesCount = np.delete(wrongExercisesCount, users_to_delete, 0)

df = df.reset_index(drop=True)

print_verb(df.shape, correctExercises.shape, wrongExercises.shape)

print_verb("The amount of users deleted in the first filter was ", len(users_to_delete))

N_USERS = len(df.Usuario)

correctExercises_minutes = correctExercises / 60
wrongExercises_minutes = wrongExercises / 60

dump_exists = os.path.isfile(dataset_prefix + str(tol) + '/' + str(tol) + '-' + str(MIN_EXERCISES) + '.csv')

if (dump_exists and not REWRITE_DUMPS):
    df_all_selected_users = pd.read_csv(dataset_prefix + str(tol) + '/' + str(tol) + '-' + str(MIN_EXERCISES) + '.csv', index_col=0)
    df_all_selected_users.fillna('', inplace=True)

    selected_users_CC = literal_eval(df_all_selected_users.loc[0][0])
    selected_users_XC = literal_eval(df_all_selected_users.loc[1][0])
    selected_users_CX = literal_eval(df_all_selected_users.loc[2][0])
    selected_users_XX = literal_eval(df_all_selected_users.loc[3][0])
    
    print_verb('Dump loaded.')
else:
    #if __name__ == '__main__':
        #__spec__ = None
        selected_users_CC = Manager().list()
        selected_users_XC = Manager().list()
        selected_users_CX = Manager().list()
        selected_users_XX = Manager().list()

        print_verb('Selecting CC users...')
        p_CC = Process(target=selectUsers, args=(selected_users_CC, tol, correctExercises_minutes, correctExercises_minutes, 'CC', N_USERS, MIN_EXERCISES,))
        p_CC.start()
        print_verb('Selecting XC users...')
        p_XC = Process(target=selectUsers, args=(selected_users_XC, tol, wrongExercises_minutes, correctExercises_minutes, 'XC', N_USERS, MIN_EXERCISES))
        p_XC.start()
        print_verb('Selecting CX users...')
        p_CX = Process(target=selectUsers, args=(selected_users_CX, tol, correctExercises_minutes, wrongExercises_minutes, 'CX', N_USERS, MIN_EXERCISES))
        p_CX.start()
        print_verb('Selecting XX users...')
        p_XX = Process(target=selectUsers, args=(selected_users_XX, tol, wrongExercises_minutes, wrongExercises_minutes, 'XX', N_USERS, MIN_EXERCISES))
        p_XX.start()

        p_CC.join()
        p_XC.join()
        p_CX.join()
        p_XX.join()

        df_all_selected_users = pd.DataFrame([selected_users_CC, selected_users_XC, selected_users_CX, selected_users_XX])
        df_all_selected_users.to_csv(dataset_prefix + str(tol) + '/' + str(tol) + '-' + str(MIN_EXERCISES) + '.csv')

        print_verb("Data stored.")


# # Calculate the time difference between the users

if __name__ == '__main__':
    __spec__ = None
    time_differences_CC = Manager().list()
    time_differences_XC = Manager().list()
    time_differences_CX = Manager().list()
    time_differences_XX = Manager().list()

    print_verb('Time differentiating CC users...')
    p_CC = Process(target=ctd.computeTimeDifferences, args=(time_differences_CC, selected_users_CC, 
                                                        correctExercises_minutes, correctExercises_minutes, 'CC', N_EXERCISES,))
    p_CC.start()
    print_verb('Time differentiating XC users...')
    p_XC = Process(target=ctd.computeTimeDifferences, args=(time_differences_XC, selected_users_XC,
                                                        wrongExercises_minutes, correctExercises_minutes, 'XC', N_EXERCISES,))
    p_XC.start()
    print_verb('Time differentiating CX users...')
    p_CX = Process(target=ctd.computeTimeDifferences, args=(time_differences_CX, selected_users_CX,
                                                        correctExercises_minutes, wrongExercises_minutes, 'CX', N_EXERCISES,))
    p_CX.start()
    print_verb('Time differentiating XX users...')
    p_XX = Process(target=ctd.computeTimeDifferences, args=(time_differences_XX, selected_users_XX,
                                                        wrongExercises_minutes, wrongExercises_minutes, 'XX', N_EXERCISES,))
    p_XX.start()

    p_CC.join()
    p_XC.join()
    p_CX.join()
    p_XX.join()
    
    print_verb('Finished.')


# ## Users in common
# 
# Finds which users are considered into the selection, without repetitions

# In[390]:


users_in_common = []

for k in range(0, len(selected_users_CC)):
    if (selected_users_CC[k][0] not in users_in_common):
        users_in_common.append(selected_users_CC[k][0])
    if (selected_users_CC[k][1] not in users_in_common):
        users_in_common.append(selected_users_CC[k][1]) 

for l in range(0, len(selected_users_XC)):
    if (selected_users_XC[l][0] not in users_in_common):
        users_in_common.append(selected_users_XC[l][0])
    if (selected_users_XC[l][1] not in users_in_common):
        users_in_common.append(selected_users_XC[l][1])

for m in range(0, len(selected_users_CX)):
    if (selected_users_CX[m][0] not in users_in_common):
        users_in_common.append(selected_users_CX[m][0])
    if (selected_users_CX[m][1] not in users_in_common):
        users_in_common.append(selected_users_CX[m][1]) 

for n in range(0, len(selected_users_XX)):
    if (selected_users_XX[n][0] not in users_in_common):
        users_in_common.append(selected_users_XX[n][0])
    if (selected_users_XX[n][1] not in users_in_common):
        users_in_common.append(selected_users_XX[n][1])

print_verb('\t Result size:', len(users_in_common))
print_verb('\t Result data:', users_in_common)


# # Join type arrays

# In[391]:


label = []
all_selected_users = []
all_time_differences = []

versions = ['CC', 'XC', 'CX', 'XX']
temp_users = [list(selected_users_CC), list(selected_users_XC), list(selected_users_CX), list(selected_users_XX)]
temp_td = [list(time_differences_CC), list(time_differences_XC), list(time_differences_CX), list(time_differences_XX)]
temp_size = [len(selected_users_CC), len(selected_users_XC), len(selected_users_CX), len(selected_users_XX)]

#for l in range(0, 4):        
#    z = temp_size.index(max(temp_size))
#    label.append(versions[z])
#    all_selected_users.append(temp_users[z])
#    all_time_differences.append(temp_td[z])
#    temp_size.pop(z) 
#    versions.pop(z) 
#    temp_users.pop(z)
#    temp_td.pop(z)

all_selected_users = temp_users
all_time_differences = temp_td
label = versions

print_verb(label)
print_verb(label[0], len(all_selected_users[0]), '||', len(selected_users_CC))
print_verb(label[1], len(all_selected_users[1]), '||', len(selected_users_XC))
print_verb(label[2], len(all_selected_users[2]), '||', len(selected_users_CX))
print_verb(label[3], len(all_selected_users[3]), '||', len(selected_users_XX))


# In[392]:


type_array = type_separation(all_selected_users, all_time_differences, tol)
print_verb('Finished')


# # Calculate user bias and pairs through the type arrays

# In[398]:


user_pairs, user_score_difference, fig, string_dump, user_interaction_percentages = generate_pairs(type_array, df.Usuario, scores, trimming, label, plot=False)

with open('user_bias_dump.txt', 'w') as filehandle:
    json.dump(string_dump, filehandle)

print_verb('Done.')
    
#print_verb('Plotting', len(user_pairs), 'pairs consisted of', len(users_in_common), 'users')
#fig.set_size_inches(20, len(fig.axes) * 6)
#fig.tight_layout()
#plt.savefig(str(tol) + '/' + str(trimming) + '-user_bias.eps')
#plt.show()


# ## Distance matrix

# matrix_size = len(users_in_common)
# 
# distance_matrix = np.ones((matrix_size, matrix_size))
# 
# for i in range(0, matrix_size):
#     user_1 = users_in_common[i]
#     for j in range(0, matrix_size):
#         user_2 = users_in_common[j]
#         for k in range(0, len(type_array)):
#             if (type_array[k][0][0] == user_1 and type_array[k][0][1] == user_2):
#                 distance_matrix[i][j] = type_array[k][2]
#                 distance_matrix[j][i] = type_array[k][2]
# 
# distance_matrix = abs(distance_matrix)
# 
# for i in range(0, matrix_size):
#     for j in range(0, matrix_size):
#         if (i == j):
#             distance_matrix[i][j] = 0
# 
# distance_matrix[np.isnan(distance_matrix)] = 1
# 
# #plt.figure(figsize=(15, 15))
# #ax = sns.heatmap(distance_matrix, annot=True, annot_kws={"size": 7}, fmt = '.2f')
# 
# #ax.set(xticklabels=users_in_common, yticklabels=users_in_common)
# #plt.show()

# ## Dendrogram

# plt.figure(figsize=(60, 28))  
# 
# #print_verb(np.shape(distance_matrix))
# 
# square = ssd.squareform(distance_matrix)
# 
# linkage = sch.linkage(square, method='single')
# 
# dend = sch.dendrogram(linkage, labels=users_in_common, leaf_rotation=90)
# 
# plt.title("Dendrogram", size=40)  
# plt.xlabel('Users', size=40)
# plt.ylabel('Distance', size=40)
# plt.xticks(size = 20)
# plt.yticks(size = 40)
# 
# plt.savefig(str(tol) + '/' + str(trimming) + '-dendrogram.png')
# 
# #plt.show()

# # Process type array and eliminate users without a score

# In[404]:


type_array = np.array(type_array)
user_score_difference = np.array(user_score_difference)

total_exercises_under_tol = type_array[:, 3]
type_array = np.reshape(type_array[:, 2], (-1, 1))
user_score_difference = np.reshape(user_score_difference, (-1, 1))

analysing_data = np.concatenate((type_array, user_score_difference), axis=1)

analysing_data = np.array(analysing_data, np.float)

user_pairs_copy = user_pairs.copy()

i = 0
while i < len(analysing_data):
    if(np.isnan(analysing_data[i,1])):
        user_pairs_copy.pop(i)
        user_interaction_percentages.pop(i)
        user_score_difference = np.delete(user_score_difference, i, 0)
        analysing_data = np.delete(analysing_data, i, 0)
        total_exercises_under_tol = np.delete(total_exercises_under_tol, i, 0)
    else:
        i = i + 1

print_verb(MIN_EXERCISES, tol, len(user_pairs_copy), len(users_in_common), 'pairs remaining.')


# ## Plot the amout of exercise by user through the whole dataset

# In[405]:


plt.figure(figsize=(10, 7))
plt.xlabel('User bias', fontsize=25)
plt.ylabel('Score difference', fontsize=25)
plt.title("Amount of exercises by user", fontsize=25)

scatter = plt.scatter(analysing_data[:, 0], analysing_data[:, 1], s=250, edgecolors = 'black', c=total_exercises_under_tol, cmap='binary')

plt.colorbar(scatter)

#for i, txt in enumerate(user_pairs_copy):
#    plt.annotate(txt, (analysing_data[i, 0], analysing_data[i, 1]))

x = np.linspace(-1, 1, 201)
y = [pow(i, 9) for i in x]
plt.plot(x, y)

axes = plt.gca()
axes.set_xlim([-1.1, 1.1])
axes.set_ylim([-1.1, 1.1])
plt.xticks(fontsize=20)

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.savefig(dataset_prefix + str(tol) + '/' + str(trimming) + '-amount_of_exercises.png', bbox_inches='tight')

#plt.show()

print_verb("TEUT:", total_exercises_under_tol.max())

plt.figure(figsize=(15, 10))
plt.hist(total_exercises_under_tol, bins=np.arange(total_exercises_under_tol.max()) + 1.5, ec='black')
plt.yscale('log')
plt.xticks(np.arange(total_exercises_under_tol.max() + 1), rotation=90)
plt.xlim([-1, total_exercises_under_tol.max() + 1])
plt.ylabel("Amount of pairs", fontsize=20)
plt.xlabel("Exercises in common", fontsize=20)
plt.savefig(str(tol) + '/' + str(trimming) + '-total_exercises_under_tol.png')

print_verb("Total number of pairs: ", len(total_exercises_under_tol))

np.savetxt("exercise_dump.txt", total_exercises_under_tol)

print_verb("Plotted.")

np.save(dataset_prefix + str(tol) + '/' + str(tol) + '-teut.npy', total_exercises_under_tol, allow_pickle=True)

#sys.exit()


# # Plot the distance from optimal curve (X^9) and separate outliers

# In[406]:


plt.figure(figsize=(10, 7))
plt.xlabel('User bias')
plt.ylabel('Course final score difference between users')
plt.title("Distance from optimal curve (X^9)")

normal_points = []
outliers = []

for i, txt in enumerate(analysing_data):
    dist = distance(analysing_data[i, 0], analysing_data[i, 1])
    if (dist > 0.25):
        plt.annotate(round(dist, 2), (analysing_data[i, 0], analysing_data[i, 1])) 
        outliers.append([analysing_data[i, 0], analysing_data[i, 1], user_pairs_copy[i][0], user_pairs_copy[i][1]])
    else:
        normal_points.append([analysing_data[i, 0], analysing_data[i, 1], user_pairs_copy[i][0], user_pairs_copy[i][1]])

#for i, txt in enumerate(user_pairs_copy):
#    plt.annotate(txt, (analysing_data[i, 0], analysing_data[i, 1]))

normal_points = np.asarray(normal_points)
outliers = np.asarray(outliers)

plt.scatter(normal_points[:, 0], normal_points[:, 1], marker='o', color='blue', picker=True)   
plt.scatter(outliers[:, 0], outliers[:, 1], marker='o', color='red', picker=True)    

x = np.linspace(-1, 1, 201)
y = [pow(i, 9) for i in x]
plt.plot(x, y)

axes = plt.gca()
axes.set_xlim([-1.1, 1.1])
axes.set_ylim([-1.1, 1.1])

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.savefig(dataset_prefix + str(tol) + '/' + str(trimming) + '-distance_from_curve.png')

#plt.show()


# ## Checking users: outliers or not

# In[407]:


FLAG = 'both'

if (FLAG == 'outliers'):
    array_being_analysed = outliers
if (FLAG == 'normals'):
    array_being_analysed = normal_points
else:
    array_being_analysed = np.concatenate((normal_points, outliers), axis=0)


# # Plot the hypotheses regions

# In[408]:


first = []
second = []
third = []
fourth = []
fifth = []
others = []

plt.figure(figsize=(11, 7))
plt.xlabel('User bias', fontsize=25)
plt.ylabel('Score difference', fontsize=25)
plt.title("Hypotheses", fontsize=25)

for i in range(0, len(array_being_analysed)):
    d = array_being_analysed[i, 0]
    sc = array_being_analysed[i, 1]

    if (d >= 0.5 and sc >= 0.25):
        first.append([d, sc])
    elif (d >= 0.5 and sc > -0.25 and sc < 0.25):
        second.append([d, sc])
    elif (d > -0.5 and d < 0.5 and sc > -0.25 and sc < 0.25):
        third.append([d, sc])
    elif (d <= -0.5 and sc > -0.25 and sc < 0.25):
        fourth.append([d, sc])
    elif (d <= -0.5 and sc <= -0.25):
        fifth.append([d, sc])
    else:
        others.append([d, sc])

first = np.asarray(first)
second = np.asarray(second)
third = np.asarray(third)
fourth = np.asarray(fourth)
fifth = np.asarray(fifth)
others = np.asarray(others)

plt.scatter(first[:, 0], first[:, 1], marker='o', color='blue', s=250, picker=True)
plt.scatter(second[:, 0], second[:, 1], marker='o', color='black', s=250, picker=True)
plt.scatter(third[:, 0], third[:, 1], marker='o', color='black', s=250, picker=True)
plt.scatter(fourth[:, 0], fourth[:, 1], marker='o', color='black', s=250, picker=True)
plt.scatter(fifth[:, 0], fifth[:, 1], marker='o', color='orange', s=250, picker=True)
plt.scatter(others[:, 0], others[:, 1], marker='o', color='grey', s=250, picker=True)

x = np.linspace(-1, 1, 201)
y = [pow(i, 9) for i in x]
plt.plot(x, y)

axes = plt.gca()
axes.set_xlim([-1.1, 1.1])
axes.set_ylim([-1.1, 1.1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.savefig(dataset_prefix + str(tol) + '/' + str(trimming) + '-hypotheses.png', bbox_inches='tight')

#plt.show()

print_verb("Checking IPs...")

ip_dump_exists = os.path.isfile(dataset_prefix + str(tol) + 'ip_addrs.npy')

if (ip_dump_exists and not REWRITE_DUMPS):
    ip_addrs = np.load(dataset_prefix + str(tol) + 'ip_addrs.npy', allow_pickle=True)
else:    
    ip_addrs = check_ip_addresses(array_being_analysed, df)
    np.save(dataset_prefix + str(tol) + 'ip_addrs.npy', ip_addrs, allow_pickle=True)

print_verb("Done.")

ips_in_common = []

for i in range(0, len(ip_addrs)):
    user_1_ips = ip_addrs[i][0]
    user_2_ips = ip_addrs[i][1]
    count_1 = ip_addrs[i][2]
    count_2 = ip_addrs[i][3]
    amount = 0
    for j in range(0, len(user_1_ips)):
        for k in range(0, len(user_2_ips)):
            if ((user_1_ips[j] == user_2_ips[k]) == True):
                amount = amount + min(count_1[j], count_2[k])
    #print_verb('-'*100)
    #print_verb('User 1 (', array_being_analysed[i][2], ',', len(user_1_ips), '): ', user_1_ips, count_1)
    #print_verb('User 2 (', array_being_analysed[i][3], ',', len(user_2_ips), '): ', user_2_ips, count_2)
    #print_verb('\nCorrelation: ', amount, np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))
    ips_in_common.append(amount)


# # Plot checked users

# In[410]:


plt.figure(figsize=(10, 7))
plt.xlabel('User bias', fontsize=25)
plt.ylabel('Course final score difference between users', fontsize=25)
plt.title("Exercises with the same IP", fontsize=25)

scatter = plt.scatter(array_being_analysed[:, 0], array_being_analysed[:, 1], edgecolors = 'black', c=ips_in_common, cmap='binary')

plt.colorbar(scatter)

#for i, txt in enumerate(ips_in_common):
    #plt.annotate((txt, (array_being_analysed[i, 2], array_being_analysed[i, 3])), (array_being_analysed[i, 0], array_being_analysed[i, 1]))

x = np.linspace(-1, 1, 201)
y = [pow(i, 9) for i in x]
plt.plot(x, y)

axes = plt.gca()
axes.set_xlim([-1.1, 1.1])
axes.set_ylim([-1.1, 1.1])
plt.xticks(fontsize=20)

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.savefig(dataset_prefix + str(tol) + '/' + str(trimming) + '-exercises_same_ip-' + FLAG + '.png')

#plt.show()


# # Calculate the material usage index

# By counting the number of events a user has, it can measure how many of them were directed towards reading the materials. This is done by dividing the count of all the events by the count of the events where the user tried to solve an exercise. However, since a user can try the same exercise multiple times, it was decided that if the user tried the exercise at least one time (be it right or wrong), then it would count as a material usage towards that exercise and that's it, no additional tries for that exercise would be considered as material usage.
# 
# Therefore, the formula is: number_of_material_usages / (number_of_material_usages + number_of_exercises_tried).
# 
# The results mean: 
# 0 -> purely exercise tryouts (fake account); 
# 
# 0,5 -> equal number of exercise tryouts and material reviews (legit user) and; 
# 
# 1 -> purely material reviews (probably a professor or a material thief)

# In[411]:


material_usage = check_material_usage(array_being_analysed, df)


# In[412]:


plt.figure(figsize=(15, 8))
plt.xlabel('User bias', fontsize=25)
plt.ylabel('Score difference', fontsize=25)
plt.title("mMIR", fontsize=25)

material_usage_str = []

for i in range(0, len(material_usage)):
    material_usage_str.append(str(min(round(material_usage[i, 0], 2), round(material_usage[i, 1], 2))))

c_intensities = []

for i in range(0, len(array_being_analysed)):
    c_intensities.append(min(material_usage[i, 0], material_usage[i, 1]))
    
scatter = plt.scatter(array_being_analysed[:, 0], array_being_analysed[:, 1], s=250, edgecolors = 'black', c=c_intensities, cmap='binary')

plt.colorbar(scatter)

material_usage_dump = []

for i, txt in enumerate(material_usage_str):
    #plt.annotate((txt, (array_being_analysed[i, 2], array_being_analysed[i, 3])), (array_being_analysed[i, 0], array_being_analysed[i, 1]))
    material_usage_dump.append('User: ' + str(array_being_analysed[i, 2]) + ' User: ' + str(array_being_analysed[i, 3]) + ' ' + txt)

with open('material_usage_dump-' + FLAG + '.txt', 'w') as filehandle:
    json.dump(material_usage_dump, filehandle)

x = np.linspace(-1, 1, 201)
y = [pow(i, 9) for i in x]
plt.plot(x, y)

axes = plt.gca()
axes.set_xlim([-1.1, 1.1])
axes.set_ylim([-1.1, 1.1])
plt.xticks(fontsize=20)

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.savefig(dataset_prefix + str(tol) + '/' + str(trimming) + '-material_usage_index-' + FLAG + '.png', bbox_inches='tight')

#plt.show()


# # Create a connection graph

# In[413]:


ip_addrs_list = []

ip_addrs_list.append(check_ip_addresses(normal_points, df))
ip_addrs_list.append(check_ip_addresses(outliers, df))

temp = []

labels_ip_both = []

for h in range(0, 2):
    ip_addrs = ip_addrs_list[h]
    for i in range(0, len(ip_addrs)):
        user_1_ips = ip_addrs[i][0]
        user_2_ips = ip_addrs[i][1]
        count_1 = ip_addrs[i][2]
        count_2 = ip_addrs[i][3]
        amount = 0
        for j in range(0, len(user_1_ips)):
            for k in range(0, len(user_2_ips)):
                if ((user_1_ips[j] == user_2_ips[k]) == True):
                    amount = amount + min(count_1[j], count_2[k])
        #print_verb('User 1 (', array_being_analysed[i][2], ',', len(user_1_ips), '): ', user_1_ips, count_1)
        #print_verb('User 2 (', array_being_analysed[i][3], ',', len(user_2_ips), '): ', user_2_ips, count_2)
        #print_verb('\nCorrelation: ', amount, np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))
        temp.append('/' + str(len(user_1_ips)) + '/' + str(len(user_2_ips)) + '/' + str(len(np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))))
    labels_ip_both.append(temp)
    
G = nx.DiGraph()

user_bias_array = analysing_data[:, 0]

labels_ip = []
i = 0

material_usage = []
material_usage_both = []
material_usage_both.append(check_material_usage(normal_points, df))
material_usage_both.append(check_material_usage(outliers, df))  

while i < len(user_pairs_copy):
    user_1 = int(user_pairs_copy[i][0])
    user_2 = int(user_pairs_copy[i][1])
    found = False

    for j in range(0, len(normal_points)):
        if (user_1 == normal_points[j][2] and user_2 == normal_points[j][3]):
            labels_ip.append(labels_ip_both[0][j])
            material_usage.append(material_usage_both[0][j])
            found = True
            i = i + 1
            break

    if (not found):   
        for k in range(0, len(outliers)):
            if (user_1 == outliers[k][2] and user_2 == outliers[k][3]):
                labels_ip.append(labels_ip_both[1][k])
                material_usage.append(material_usage_both[1][k])
                i = i + 1
                break

""" node_colors = {}

for i in range(0, len(material_usage)):           
    user_1 = int(user_pairs_copy[i][0])
    user_2 = int(user_pairs_copy[i][1])
    if (material_usage[i][0] == 0):
        node_colors[user_1] = 'red'
    else:
        node_colors[user_1] = 'blue'
    if (material_usage[i][1] == 0):
        node_colors[user_2] = 'red'
    else:
        node_colors[user_2] = 'blue'

node_color_array = []

labels = {}

for i in range(0, len(user_pairs_copy)):
    user_1 = int(user_pairs_copy[i][0])
    user_2 = int(user_pairs_copy[i][1])
    G.add_node(user_1)
    G.add_node(user_2)

    temp = abs(round(user_bias_array[i], 2))

    if (user_bias_array[i] >= 0):
        G.add_edge(user_1, user_2, width=temp * 10)
        labels[user_1, user_2] = str(temp) + str(labels_ip[i])
    else:
        G.add_edge(user_2, user_1, width=temp * 10)
        labels[user_2, user_1] = str(temp) + str(labels_ip[i])

for node in G:
    node_color_array.append(node_colors[node])

pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args="-Gnodesep=5")


# # Plot connection graph

plt.close('all')

plt.figure(figsize=(50, 50))

width = [G[u][v]['width'] for u,v in G.edges()]      

nx.draw(G, pos, node_color=node_color_array)

nx.draw_networkx_edges(G, pos, width=width)

nx.draw_networkx_labels(G, pos, font_size=10)

text = nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red')

for _,t in text.items():
    t.set_rotation('vertical')

plt.savefig(str(tol) + '/' + str(trimming) + '-connection-graph' + '.png')
"""

temp = []

for i, txt in enumerate(analysing_data):
    temp.append([analysing_data[i, 0], analysing_data[i, 1], user_pairs_copy[i][0], user_pairs_copy[i][1]])

ip_addrs_temp = check_ip_addresses(temp, df)

ips_in_common_temp = []

for i in range(0, len(ip_addrs_temp)):
    user_1_ips = ip_addrs_temp[i][0]
    user_2_ips = ip_addrs_temp[i][1]
    amount = 0
    for j in range(0, len(user_1_ips)):
        for k in range(0, len(user_2_ips)):
            if (user_1_ips[j] == user_2_ips[k]):
                amount = amount + 1
    #print_verb('Total:', len(user_1_ips), len(user_2_ips), amount)
    #print_verb('-'*100)
    #print_verb('User 1 (', array_being_analysed[i][2], ',', len(user_1_ips), '): ', user_1_ips, count_1)
    #print_verb('User 2 (', array_being_analysed[i][3], ',', len(user_2_ips), '): ', user_2_ips, count_2)
    #print_verb('\nCorrelation: ', amount, np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))
    total = len(user_1_ips) + len(user_2_ips)
    ips_in_common_temp.append(amount / (total - amount))

uip = np.asarray(user_interaction_percentages)
mu = np.asarray(material_usage)

CC = uip[:, 0]
XC = uip[:, 1]
XX = uip[:, 2]
MIR = [min(pair[0], pair[1]) for pair in mu]
ScoreDif = user_score_difference[:, 0]
#IPs = np.asarray([i / j  for i, j in zip(ips_in_common_temp, total_exercises_under_tol)])
IPs = np.asarray(ips_in_common_temp)
UB = user_bias_array
#TE = total_exercises_under_tol

ix = user_bias_array<0

UB[ix] = -UB[ix]
ScoreDif[ix] = -ScoreDif[ix]

MIR = np.asarray(MIR)

dict_var = {'CC':CC, 'XC':XC, 'User Bias':UB, 'Minimal MIR':MIR, 'Score Diff':ScoreDif, 
            'IPs in common':IPs}
#dict_var = {'CC':CC, 'XX':XX, 'User Bias':UB, 'Minimal MIR':MIR, 'Score Diff':ScoreDif, 
#            'IPs in common':IPs}

for key, value in dict_var.items():
    print_verb(key, len(value), value.shape)
    
x = pd.DataFrame.from_dict(dict_var)

# Get column names first
names = x.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
x.index = user_pairs_copy
x=x[:-1]

scaled_x = scaler.fit_transform(x)
scaled_x = pd.DataFrame(scaled_x, columns=names)

# # Clustering

if (TIME_LIMIT):
    full_data_df = pd.read_pickle('final_df-' + str(tol) + '.pkl')
    idxs = x.index
    y = full_data_df.loc[idxs]['Labels']
    x['Labels'] = y

else:
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=0)

    kmeans.fit_predict(scaled_x)

    print_verb(kmeans.labels_)
    print_verb(kmeans.cluster_centers_)

    scaler.inverse_transform(kmeans.cluster_centers_)

    y = kmeans.labels_

    x['Labels'] = y

    x.to_pickle('final_df-' + str(tol) + '.pkl')
    sys.exit()

harvester = len(y[y == 0]) 
collaboration = len(y[y == 1]) 

print_verb('In this dataset, there are', harvester, 'harvester type copies and', collaboration, 'collaboration type copies.')


# ## Supervised learning

print_verb("There are", collaboration, "1's and", collaboration, "0's as labels.")
prop_1 = collaboration / len(y)
prop_0 = harvester / len(y)
print("The proportion of 1's and 0's is, repectively:", prop_1, prop_0)

from sklearn import svm
from sklearn.model_selection import GridSearchCV

class_weights = {}
class_weights[0] = prop_0
class_weights[1] = prop_1

print_verb(class_weights)

NUM_CV = 10

X = scaled_x

# TODO
# Sometimes the clusters will be inverted. There should be a method capable of identifying them correctly.

if (tol == 2):
    y = 1 - y

C = np.linspace(-5, 11, 9)
C = [pow(2, i) for i in C] #2^[-5, -3, ... , 10]
gamma = np.linspace(-1, 15, 9)
gamma = [pow(2, i) for i in gamma] #2^[-1, 1, ... , 15]

# In[422]:


# Set up possible values of parameters to optimize over
          
p_grid = [{'kernel': ['rbf'], 'gamma': gamma, 'C': C},
        {'kernel': ['linear'], 'C': C}]
svc = svm.SVC(class_weight=class_weights, probability=True)
clf = GridSearchCV(svc, param_grid=p_grid, cv=NUM_CV, iid=False, refit=True)
print_verb(clf)

import pickle

filename = 'finalized_model-' + str(tol) + '.sav'

if(NEW_DATASET):
    # load the model from disk
    loaded_model = pickle.load(open(filename, 'rb'))
    result = loaded_model.score(X, y)
    print("Result from loading model:", result)

    pred = np.asarray(loaded_model.predict_proba(X))

    pred = pred[range(len(pred)), y]

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=0) 
    auc = metrics.auc(fpr, tpr)
    # plot the roc curve for the model
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', label='AUC = ' + str(auc))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()

    plt.savefig(str(tol) + '/' + str(trimming) + '-roc_auc-0.png')

    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1) 
    auc = metrics.auc(fpr, tpr)
    # plot the roc curve for the model
    plt.figure()
    plt.plot(fpr, tpr, linestyle='--', label='AUC = ' + str(auc))
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.savefig(str(tol) + '/' + str(trimming) + '-roc_auc-1.png')
else:
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    from sklearn.model_selection import cross_val_score, cross_val_predict

    if (os.path.isfile("results_dump.csv")):
        os.remove("results_dump.csv")

    if (os.path.isfile("X_dump.csv")):
        os.remove("X_dump.csv")
        
    if (os.path.isfile("sv_dump.csv")):
        os.remove("sv_dump.csv")
        
    def my_scoring(estimator, X, y):
        results = estimator.predict_proba(X)
        svs = estimator.best_estimator_.support_vectors_
        
        df_results = pd.DataFrame(results)
        with open('results_dump.csv', 'a') as f:
            df_results.to_csv(f, header=False, index=False)
        
        df_X = pd.DataFrame(X)
        with open('X_dump.csv', 'a') as f:
            df_X.to_csv(f, header=False, index=False)
            
        df_sv = pd.DataFrame(svs)
        with open('sv_dump.csv', 'a') as f:
            df_sv.to_csv(f, header=False, index=False)
        
        #plt.figure()
        #plt.hist(results[:, 0], bins=len(results))
        #plt.show()
        
        return estimator.best_score_

    # This CV is done to estimate the performance of the algorithm on unseen data and also get rid of the bias introduced 
    # by the GridSearch. Therefore, if the results are good, we can run the GridSearch again and fixate the hyperparameters.
    #scores = cross_val_score(clf, X, y, cv=NUM_CV, scoring=my_scoring)
    #print_verb("This model can differentiate harvesters (0) from collaborators (1) with an accuracy and standard deviation of, respectively: %0.3f (+/- %0.3f)" % (scores.mean() * 100, scores.std() * 200))

    # save the model to disk
    clf.fit(X, y)
    print_verb(clf)
    pickle.dump(clf.best_estimator_, open(filename, 'wb'))

# In[423]:

sys.exit()


df_results = pd.read_csv('results_dump.csv', index_col=0, header=None)
df_results.fillna('', inplace=True)

df_X = pd.read_csv('X_dump.csv', index_col=0, header=None)
df_X.fillna('', inplace=True)


# In[424]:


print_verb(np.shape(df_results), np.shape(df_X))


# In[425]:


df_classes = pd.concat([df_results.reset_index(), df_X.reset_index()], axis=1)
df_classes.columns = ['0', '1', 'CC', 'XC', 'XX', 'User Bias', 'Minimal MIR', 'Score Diff', 'IPs in common']
df_classes['Clustering labels'] = kmeans.labels_
#df_classes = df_classes.sort_values(by =['CC', 'XC', 'User Bias', 'Minimal MIR', 'Score Diff', 'IPs in common'])
scaled_x.index = x.index
#scaled_x = scaled_x.sort_values(by =['CC', 'XC', 'User Bias', 'Minimal MIR', 'Score Diff', 'IPs in common'])
df_classes.index = scaled_x.index
indexes = x.index
indexes = sorted(indexes,key=lambda x: x[0])
df_classes = df_classes.reindex(index=indexes)


# In[426]:


df_classes


# In[427]:


labels = []

for prob in df_classes['0']:
    if (prob > 0.50):
        labels.append(0)
    else:
        labels.append(1) 

df_classes['Classification labels'] = labels


# In[428]:


total = []

indexes = np.asarray(indexes)

count_x = np.unique(indexes, return_counts=True)

sum(count_x[1] > 1)

count_x[0][count_x[1][:] == 2]


# In[429]:


x = np.array([1, 2, 5, 10])
y = np.array([1, 0.97, 0.95, 0.96]) # Effectively y = x**2
e = np.array([0, 0.15, 0.22, 0.13])

plt.errorbar(x, y, e, linestyle='None', marker='^')

plt.xlabel('Time windows', fontsize=15)
plt.ylabel('Average score', fontsize=15)

plt.yticks(fontsize=15)
plt.xticks(np.linspace(0, 10, 11), fontsize=15)

plt.show()


# In[430]:


df_classes.corr()


# In[431]:


unique_users = []

for i in range(len(indexes)):
    for user in indexes[i]:
        if user not in unique_users:
            unique_users.append(user)


# In[432]:


labels_per_user = []

for user in unique_users:
    labels = []
    for i in range(len(indexes)):
        if (user == indexes[i][0]):
            labels.append(df_classes["Clustering labels"][i])
        elif (user == indexes[i][1]):
            labels.append(df_classes["Clustering labels"][i])
    labels_per_user.append(labels)
            
shifting_count = [np.unique(j, return_counts=True) for j in labels_per_user]


# In[433]:


for user, shifts in zip(unique_users, shifting_count):
    #print_verb(shifts[1])
    if len(shifts[1]) > 1:
        print_verb(user, len(shifts[1]))


# In[434]:


N = 4

#ten = [92, 147, 437, 864] #users
#twenty = [69, 71, 79, 137] #users
ten = [53, 74, 271, 769] # pairs
twenty = [37, 38, 44, 77] # pairs

difference = [a - b for a, b in zip(ten, twenty)]
ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

plt.figure(figsize=(10, 7))

p2 = plt.bar(ind, twenty, width)
p1 = plt.bar(ind, difference, width, bottom=twenty)

plt.ylabel('Amount of pairs', fontsize=25)
plt.xlabel('Time window (minutes)', fontsize=25)
plt.xticks(ind, ('1', '2', '5', '10'), fontsize=20)
plt.yticks(fontsize=20)
plt.legend((p1[0], p2[0]), ('10 exercises', '20 exercises'), prop={'size': 25})

plt.savefig('user-selection.eps')
plt.show()


# In[ ]:




