#!/usr/bin/env python
# coding: utf-8

# # Cheaters demise code

# The previous code defined the clusters of suspicious users, which were characterized as those who didn't watch (almost) any videos and documents and got all the answers right. These are thought to be users who deal with fake and real accounts, counting errors on the fakes and taking the right answers on the real ones.
# 
# Since there is a defined cluster of suspicious users, the following code will try to identify the activity of these users and see if they can be considered as cheaters or not, depending on timing and improvement. By taking into account that timing would be equal to the time difference between an answer from one user and the other, there may be a correlation between them, since after finding the real answer on a fake account, the user would input it into his real account in a short period. Also, improvement is a metric that can differentiate between a user that coincidently did it right after another and a cheater, by measuring if the user got the wrong answer or not. If it did get the wrong answer, than he's not using a fake account.
# 
# One course of action is to create a graph of exercise (i) by user(j) and the time it took the user to finish it.

# ## Dependencies

# In[1]:


# python3 -m pip install matplotlib scipy networkx seaborn sklearn pandas sympy


# ## Clear environment

# In[114]:


#get_ipython().run_line_magic('reset', '-f')


# In[115]:


#get_ipython().run_line_magic('matplotlib', 'inline')


#  # Import all definitions and libraries

# In[124]:


import importlib
import Definitions
import ctd
importlib.reload(Definitions)
importlib.reload(ctd)

from Definitions import *


# # Read scores and user data

# In[117]:


df_total = readDataFile("eventos_final.json")
scores_csv = pd.read_csv("UAMx_Android301x_1T2015_grade_report_2015-04-21-1145_sanitized.csv")

# In[118]:


df = df_total#.head(200)

print('Users being processed:', len(df))

# In[119]:


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

print("Logging scores...")

for i in range(1, len(df.Usuario)):
    try:
        score = scores_csv.loc[scores_csv['id'] == int(df.Usuario[i])]['grade'].item()
        scores.append((int(df.Usuario[i]), score))
    except:                      
        scores.append((int(df.Usuario[i]), np.nan))

    #clear()
    #print('Scores:', math.ceil(i*100/len(df.Usuario)), '% done')

print("Finished logging scores.")
print("Logging exercises...")

for i in range(0, len(df.Usuario)):
    for j in range(0, len(df.Eventos[i])):
        if (df.Eventos[i][j]['evento'] == 'problem_check'):
            #print('\t problem_check')
            if (df.Eventos[i][j]['resultados'] != []):
                #print('\t \t results')
                # convert date time to epoch time for a better comparison
                time_split = df.Eventos[i][j]['tiempo'].split('T')
                time_tuple = time.strptime(time_split[0] + ' ' + time_split[1][:8], date_format)
                time_epoch = time.mktime(time_tuple)   
                if(df.Eventos[i][j]['resultados'][0]['correcto'] == 'True'):
                    #print('\t \t \t right')
                    correctExercises[i][int(df.Eventos[i][j]['id_problema']) - 1] = time_epoch
                    correctExercisesCount[i] = correctExercisesCount[i] + 1
                elif(df.Eventos[i][j]['resultados'][0]['correcto'] == 'False'):
                    #print('\t \t \t wrong')
                    wrongExercises[i][int(df.Eventos[i][j]['id_problema']) - 1] = time_epoch
                    num_intentos = int(df.Eventos[i][j]['num_intentos'])
                    wrongExercisesCount[i] = (wrongExercisesCount[i] - num_intentos + 1) + num_intentos
    #clear()
    #print('Exercises:', math.ceil(i*100/N_USERS), '% done')

print('Ended logging.')

N_USERS = len(df)


# In[122]:


N_USERS = len(df.Usuario)

correctExercises_minutes = correctExercises / 60
wrongExercises_minutes = wrongExercises / 60


# # Select users

# In[ ]:


dump_exists = os.path.isfile(str(tol) + '/' + str(tol) + '.npy')

if (dump_exists):    
    print(hours() + 'Loading type array file...')

    type_array = np.load(str(tol) + '/' + str(tol) + '.npy', allow_pickle=True)
    percentile_under_tol = np.load(str(tol) + '/' + str(tol) + 'percentile_under_tol.npy', allow_pickle=True)

    print('Dump loaded.')
else:
    additional = N_USERS % 4
    const = int(((N_USERS - additional) / 4))

    lims = [[0, const - 1], [const - 1, (2 * const) - 1], [(2 * const) - 1, (3 * const) - 1], [(3 * const) - 1, (4 * const) - 1 + additional]]

    print('Additional:', additional, 'Const:', const, 'Lims:', lims)

    #input("Is the data correct?")

    print('Dividing threads...')
    print(lims)

    print(hours() + 'Starting processes...')
    print('Selecting CC users...')
    interactions_CC = split_work(countInteractions, tol, lims, correctExercises_minutes, correctExercises_minutes, 'CC', N_USERS)
    print('Selecting XC users...')
    interactions_XC = split_work(countInteractions_XC, tol, lims, wrongExercises_minutes, correctExercises_minutes, 'XC', N_USERS)
    print('Selecting CX users...')
    interactions_CX = split_work(countInteractions_CX, tol, lims, correctExercises_minutes, wrongExercises_minutes, 'CX', N_USERS)
    print('Selecting XX users...')
    interactions_XX = split_work(countInteractions, tol, lims, wrongExercises_minutes, wrongExercises_minutes, 'XX', N_USERS)

    print(hours() + 'Joining interactions...')
    [type_array, percentile_under_tol] = join_interaction(interactions_CC, interactions_XC, interactions_CX, interactions_XX, len(df.Usuario), correctExercises_minutes, wrongExercises_minutes)
    print(hours() + 'Finished joining interactions')
    
    print(hours() + 'Saving type array file...')
    temp = np.asarray(type_array)
    np.save(str(tol) + '/' + str(tol) + '.npy', temp, allow_pickle=True)
    temp = np.asarray(percentile_under_tol)
    np.save(str(tol) + '/' + str(tol) + 'percentile_under_tol.npy', temp, allow_pickle=True)

    print(hours() + 'Data stored.')


# In[ ]:


# # Calculate user bias and pairs through the type arrays

# In[ ]:


label = ['CC', 'XC', 'CX', 'XX']

user_pairs, user_score_difference, fig, string_dump, user_interaction_percentages = generate_pairs(type_array, df.Usuario, scores, trimming, label, plot=False)

with open('user_bias_dump.txt', 'w') as filehandle:
    json.dump(string_dump, filehandle)

print('Done.')

#fig.set_size_inches(20, len(fig.axes) * 6)
#fig.tight_layout()
#plt.savefig(str(tol) + '/' + str(trimming) + '-user_bias.eps')
#plt.show()


# # Eliminate users without a score

# In[ ]:


type_array_bkp = np.array(type_array)


# In[ ]:


type_array = type_array_bkp
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

print(tol, len(user_pairs_copy), 'pairs remaining.')

ip_addrs = check_ip_addresses(array_being_analysed, df)

ips_in_common = []

for i in range(0, len(ip_addrs)):
    user_1_ips = ip_addrs[i][0]
    user_2_ips = ip_addrs[i][1]
    count_1 = ip_addrs[i][2]
    count_2 = ip_addrs[i][3]
    amount = 0
    for j in range(0, len(user_1_ips)):
        for k in range(0, len(user_2_ips)):
            if (user_1_ips[j] == user_2_ips[k]):
                amount = amount + min(count_1[j], count_2[k])
    #print('-'*100)
    #print('User 1 (', array_being_analysed[i][2], ',', len(user_1_ips), '): ', user_1_ips, count_1)
    #print('User 2 (', array_being_analysed[i][3], ',', len(user_2_ips), '): ', user_2_ips, count_2)
    #print('\nCorrelation: ', amount, np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))
    ips_in_common.append(amount)

plt.figure(figsize=(10, 7))
plt.xlabel('User bias', fontsize=25)
plt.ylabel('Course final score difference between users', fontsize=25)
plt.title("Exercises with the same IP", fontsize=25)

scatter = plt.scatter(array_being_analysed[:, 0], array_being_analysed[:, 1], edgecolors = 'black', c=ips_in_common, cmap='binary')

plt.colorbar(scatter)

i = 0
while i < len(ips_in_common):
    if(ips_in_common[i] < 1 or ips_in_common[i] > 10):
        user_pairs_copy.pop(i)
        user_interaction_percentages.pop(i)
        user_score_difference = np.delete(user_score_difference, i, 0)
        analysing_data = np.delete(analysing_data, i, 0)
        total_exercises_under_tol = np.delete(total_exercises_under_tol, i, 0)
        percentile_under_tol = np.delete(percentile_under_tol, i, 0)
    else:
        i = i + 1

plt.hist(percentile_under_tol, bins=100)
plt.yscale('log')
plt.savefig(str(tol) + '/' + str(trimming) + '-percentile_under_tol.png')
plt.show()
# ## Plot the amout of exercise by user through the whole dataset

# In[ ]:


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
plt.savefig(str(tol) + '/' + str(trimming) + '-amount_of_exercises.png', bbox_inches='tight')

#plt.show()

plt.figure(figsize=(15, 10))
plt.hist(total_exercises_under_tol, bins=np.arange(total_exercises_under_tol.max()) - 0.5, ec='black')
plt.yscale('log')
plt.xticks(range(total_exercises_under_tol.max() - 1), rotation=90)
plt.xlim([-1, total_exercises_under_tol.max() - 1])
plt.savefig(str(tol) + '/' + str(trimming) + '-total_exercises_under_tol.png')

np.savetxt("exercise_dump.txt", total_exercises_under_tol)

# # Plot the distance from optimal curve (X^9) and separate outliers

# In[ ]:


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

normal_points = np.asarray(normal_points)
outliers = np.asarray(outliers)

if (len(normal_points) > 0):
    plt.scatter(normal_points[:, 0], normal_points[:, 1], marker='o', color='blue', picker=True)   
if (len(outliers) > 0):
    plt.scatter(outliers[:, 0], outliers[:, 1], marker='o', color='red', picker=True)    

x = np.linspace(-1, 1, 201)
y = [pow(i, 9) for i in x]
plt.plot(x, y)

axes = plt.gca()
axes.set_xlim([-1.1, 1.1])
axes.set_ylim([-1.1, 1.1])

plt.grid(color='grey', linestyle='--', linewidth=.5)

plt.savefig(str(tol) + '/' + str(trimming) + '-distance_from_curve.png')

#plt.show()


# ## Checking users: outliers or not

# In[ ]:


FLAG = 'both'

if (FLAG == 'outliers'):
    array_being_analysed = outliers
elif (FLAG == 'normals'):
    array_being_analysed = normal_points
else:
    array_being_analysed = np.concatenate((normal_points, outliers), axis=0)


# # Plot the hypotheses regions

# In[ ]:


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

if (len(first) > 0):
    plt.scatter(first[:, 0], first[:, 1], marker='o', color='blue', s=250, picker=True)
if (len(second) > 0):
    plt.scatter(second[:, 0], second[:, 1], marker='o', color='black', s=250, picker=True)
if (len(third) > 0):
    plt.scatter(third[:, 0], third[:, 1], marker='o', color='black', s=250, picker=True)
if (len(fourth) > 0):
    plt.scatter(fourth[:, 0], fourth[:, 1], marker='o', color='black', s=250, picker=True)
if (len(fifth) > 0):
    plt.scatter(fifth[:, 0], fifth[:, 1], marker='o', color='orange', s=250, picker=True)
if (len(others) > 0):
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

plt.savefig(str(tol) + '/' + str(trimming) + '-hypotheses.png', bbox_inches='tight')

#plt.show()


# In[ ]:




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

plt.savefig(str(tol) + '/' + str(trimming) + '-exercises_same_ip-' + FLAG + '.png')

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

# In[ ]:


material_usage = check_material_usage(array_being_analysed, df)


# In[ ]:


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

plt.savefig(str(tol) + '/' + str(trimming) + '-material_usage_index-' + FLAG + '.png', bbox_inches='tight')

#plt.show()


# # Sort material usage

# In[ ]:


material_usage_sorted = []
material_usage_both = []
material_usage_both.append(check_material_usage(normal_points, df))
material_usage_both.append(check_material_usage(outliers, df))  

i = 0

while i < len(user_pairs_copy):
    user_1 = int(user_pairs_copy[i][0])
    user_2 = int(user_pairs_copy[i][1])
    found = False

    for j in range(0, len(normal_points)):
        if (user_1 == normal_points[j][2] and user_2 == normal_points[j][3]):
            material_usage_sorted.append(material_usage_both[0][j])
            found = True
            i = i + 1
            break

    if (not found):   
        for k in range(0, len(outliers)):
            if (user_1 == outliers[k][2] and user_2 == outliers[k][3]):
                material_usage_sorted.append(material_usage_both[1][k])
                i = i + 1
                break


# # Clustering

# In[ ]:


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
    #print('Total:', len(user_1_ips), len(user_2_ips), amount)
    #print('-'*100)
    #print('User 1 (', array_being_analysed[i][2], ',', len(user_1_ips), '): ', user_1_ips, count_1)
    #print('User 2 (', array_being_analysed[i][3], ',', len(user_2_ips), '): ', user_2_ips, count_2)
    #print('\nCorrelation: ', amount, np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))
    total = len(user_1_ips) + len(user_2_ips)
    ips_in_common_temp.append(amount / (total - amount))


# In[ ]:


uip = np.asarray(user_interaction_percentages)
mu = np.asarray(material_usage_sorted)

CC = uip[:, 0]
XC = uip[:, 1]
XX = uip[:, 2]
MIR = [min(pair[0], pair[1]) for pair in mu]
ScoreDif = user_score_difference[:, 0]
#IPs = np.asarray([i / j  for i, j in zip(ips_in_common_temp, total_exercises_under_tol)])
IPs = np.asarray(ips_in_common_temp)
UB = analysing_data[:, 0]
#TE = total_exercises_under_tol

ix = analysing_data[:, 0]<0

UB[ix] = -UB[ix]
ScoreDif[ix] = -ScoreDif[ix]

MIR = np.asarray(MIR)

dict_var = {'CC':CC, 'XC':XC, 'User Bias':UB, 'Minimal MIR':MIR, 'Score Diff':ScoreDif, 
            'IPs in common':IPs}
#dict_var = {'CC':CC, 'XX':XX, 'User Bias':UB, 'Minimal MIR':MIR, 'Score Diff':ScoreDif, 
#            'IPs in common':IPs}

for key, value in dict_var.items():
    print(key, len(value), value.shape)
    
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

pd.set_option('display.max_rows', len(x))
#x.drop(x.tail(1).index,inplace=True)
x.shape


# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit_predict(scaled_x)

print(kmeans.labels_)
print(kmeans.cluster_centers_)

x['Labels'] = kmeans.labels_
x


# In[ ]:


scaler.inverse_transform(kmeans.cluster_centers_)


# In[ ]:


harvester = len(kmeans.labels_[kmeans.labels_ == 0]) 
person = len(kmeans.labels_[kmeans.labels_ == 1]) 

print('In this dataset, there are', harvester, 'harvester type copies and', person, 'person type copies.')


# ## Supervised learning

# In[ ]:


print("There are", len(x['Labels'][x['Labels'] == 1]), "1's and", len(x['Labels'][x['Labels'] == 0]), "0's as labels.")
prop_1 = len(x['Labels'][x['Labels'] == 1]) / len(x['Labels'])
prop_0 = len(x['Labels'][x['Labels'] == 0]) / len(x['Labels'])
print("The proportion of 1's and 0's is, repectively:", prop_1, prop_0)


# In[ ]:


from sklearn import svm
from sklearn.model_selection import GridSearchCV

class_weights = {}
class_weights[0] = prop_0
class_weights[1] = prop_1

print(class_weights)

NUM_CV = 10

X = scaled_x
y = x['Labels']

C = np.linspace(-5, 11, 9)
C = [pow(2, i) for i in C] #2^[-5, -3, ... , 10]
gamma = np.linspace(-1, 15, 9)
gamma = [pow(2, i) for i in gamma] #2^[-1, 1, ... , 15]


# In[ ]:


# Set up possible values of parameters to optimize over
          
p_grid = {'C': C, 'gamma': gamma}
    
svc = svm.SVC(kernel = 'linear', class_weight=class_weights, probability=True)
clf = GridSearchCV(svc, param_grid=p_grid, cv=NUM_CV, iid=False, refit=True)
print(clf)

from sklearn.model_selection import cross_val_score, cross_val_predict

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

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

scores = cross_val_score(clf, X, y, cv=NUM_CV, scoring=my_scoring)

print("This model can differentiate harvesters (0) from collaborators (1) with an accuracy and standard deviation of, respectively: %0.3f (+/- %0.3f)" % (scores.mean() * 100, scores.std() * 200))


# In[ ]:


df_results = pd.read_csv('results_dump.csv', index_col=0, header=None)
df_results.fillna('', inplace=True)

df_X = pd.read_csv('X_dump.csv', index_col=0, header=None)
df_X.fillna('', inplace=True)


# In[ ]:


print(np.shape(df_results), np.shape(df_X))


# In[ ]:


df_classes = pd.concat([df_results.reset_index(), df_X.reset_index()], axis=1)
df_classes.columns = ['0', '1', 'CC', 'XC', 'User Bias', 'Minimal MIR', 'Score Diff', 'IPs in common']
df_classes['Clustering labels'] = kmeans.labels_
#df_classes = df_classes.sort_values(by =['CC', 'XC', 'User Bias', 'Minimal MIR', 'Score Diff', 'IPs in common'])
scaled_x.index = x.index
#scaled_x = scaled_x.sort_values(by =['CC', 'XC', 'User Bias', 'Minimal MIR', 'Score Diff', 'IPs in common'])
df_classes.index = scaled_x.index
indexes = x.index
indexes = sorted(indexes,key=lambda x: x[0])
df_classes = df_classes.reindex(index=indexes)


# In[ ]:


labels = []

for prob in df_classes['0']:
    if (prob > 0.50):
        labels.append(0)
    else:
        labels.append(1) 

df_classes['Classification labels'] = labels


# In[ ]:


unique_users = []

for i in range(len(indexes)):
    for user in indexes[i]:
        if user not in unique_users:
            unique_users.append(user)


# In[ ]:


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


# # Shifting users

# In[ ]:


for user, shifts in zip(unique_users, shifting_count):
    #print(shifts[1])
    if len(shifts[1]) > 1:
        print(user, len(shifts[1]))


# In[ ]:




