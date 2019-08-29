#!/usr/bin/env python
# coding: utf-8

# # Cheaters demise code

# The previous code defined the clusters of suspicious users, which were characterized as those who didn't watch (almost) any videos and 
# documents and got all the answers right. These are thought to be users who deal with fake and real accounts, counting errors on the fakes 
# and taking the right answers on the real ones.
# Since there is a defined cluster of suspicious users, the following code will try to identify the activity of these users and see if they 
# can be considered as cheaters or not, depending on timing and improvement. By taking into account that timing would be equal to the time 
# difference between an answer from one user and the other, there may be a correlation between them, since after finding the real answer on 
# a fake account, the user would input it into his real account in a short period. Also, improvement is a metric that can differentiate 
# between a user that coincidently did it right after another and a cheater, by measuring if the user got the wrong answer or not. If it did 
# get the wrong answer, than he's not using a fake account.
# 
# One course of action is to create a graph of exercise (i) by user(j) and the time it took the user to finish it.

if __name__ == '__main__':
    from Definitions import *
    def main():
        # ## Import the data

        # # Read scores and user data

        df = readDataFile("eventos_final.json")
        scores_csv = pd.read_csv("UAMx_Android301x_1T2015_grade_report_2015-04-21-1145_sanitized.csv")
        ids_sospechosos = pd.read_csv("UserIDClusterSospechoso.csv")

        ## Use the data from the suspicious cluster (disabled)
        #new_df_data = []
        #
        #for i in range(0, len(ids_sospechosos.Usuario)):
        #    df.Usuario[df.Usuario == str(ids_sospechosos.Usuario[i])]
        #    new_df_data.append(df.loc[df.Usuario[df.Usuario == str(ids_sospechosos.Usuario[i])].index[0]])
        #    
        #df = pd.DataFrame(new_df_data, columns = ['Eventos', 'Usuario'])
        #df = df.reset_index(drop=True)
        #
        #cluster_tag = 'cluster'
        #
        #print('Ended.')

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

        # Cutting down the size of the arrays

        print(df.shape, correctExercises.shape, wrongExercises.shape)

        users_to_delete = []

        for i in range(0, N_USERS):
            nbr_attempts = wrongExercisesCount[i] + correctExercisesCount[i]
            if (nbr_attempts < 10):
                df = df.drop([i])
                users_to_delete.append(i)

        correctExercises = np.delete(correctExercises, users_to_delete, 0)
        wrongExercises = np.delete(wrongExercises, users_to_delete, 0)
        correctExercisesCount = np.delete(correctExercisesCount, users_to_delete, 0)
        wrongExercisesCount = np.delete(wrongExercisesCount, users_to_delete, 0)
            
        df = df.reset_index(drop=True)

        print(df.shape, correctExercises.shape, wrongExercises.shape)

        N_USERS = len(df.Usuario)

        correctExercises_minutes = correctExercises / 60
        wrongExercises_minutes = wrongExercises / 60

        dump_exists = os.path.isfile(str(tol) + '/' + str(tol) + '.csv')

        if (dump_exists):
            df_all_selected_users = pd.read_csv(str(tol) + '/' + str(tol) + '.csv', index_col=0)
            df_all_selected_users.fillna('', inplace=True)

            selected_users_CC = literal_eval(df_all_selected_users.loc[0][0])
            selected_users_XC = literal_eval(df_all_selected_users.loc[1][0])
            selected_users_CX = literal_eval(df_all_selected_users.loc[2][0])
            selected_users_XX = literal_eval(df_all_selected_users.loc[3][0])
        else:
            selected_users_CC = Manager().list()
            selected_users_XC = Manager().list()
            selected_users_CX = Manager().list()
            selected_users_XX = Manager().list()

            print('Selecting CC users...')
            p_CC = Process(target=selectUsers, args=(selected_users_CC, tol, correctExercises_minutes, correctExercises_minutes, 'CC',))
            p_CC.start()
            print('Selecting XC users...')
            p_XC = Process(target=selectUsers, args=(selected_users_XC, tol, wrongExercises_minutes, correctExercises_minutes, 'XC',))
            p_XC.start()
            print('Selecting CX users...')
            p_CX = Process(target=selectUsers, args=(selected_users_CX, tol, correctExercises_minutes, wrongExercises_minutes, 'CX',))
            p_CX.start()
            print('Selecting XX users...')
            p_XX = Process(target=selectUsers, args=(selected_users_XX, tol, wrongExercises_minutes, wrongExercises_minutes, 'XX',))
            p_XX.start()

            p_CC.join()
            p_XC.join()
            p_CX.join()
            p_XX.join()

            df_all_selected_users = pd.DataFrame([selected_users_CC, selected_users_XC, selected_users_CX, selected_users_XX])
            df_all_selected_users.to_csv(str(tol) + '/' + str(tol) + '.csv')

            print("Data stored.")

        # Calculate the time difference between the users

        time_differences_CC = Manager().list()
        time_differences_XC = Manager().list()
        time_differences_CX = Manager().list()
        time_differences_XX = Manager().list()

        print('Time differentiating CC users...')
        p_CC = Process(target=ctd.computeTimeDifferences, args=(time_differences_CC, selected_users_CC, 
                                                            correctExercises_minutes, correctExercises_minutes, 'CC', N_EXERCISES,))
        p_CC.start()
        print('Time differentiating XC users...')
        p_XC = Process(target=ctd.computeTimeDifferences, args=(time_differences_XC, selected_users_XC,
                                                            wrongExercises_minutes, correctExercises_minutes, 'XC', N_EXERCISES,))
        p_XC.start()
        print('Time differentiating CX users...')
        p_CX = Process(target=ctd.computeTimeDifferences, args=(time_differences_CX, selected_users_CX,
                                                            correctExercises_minutes, wrongExercises_minutes, 'CX', N_EXERCISES,))
        p_CX.start()
        print('Time differentiating XX users...')
        p_XX = Process(target=ctd.computeTimeDifferences, args=(time_differences_XX, selected_users_XX,
                                                            wrongExercises_minutes, wrongExercises_minutes, 'XX', N_EXERCISES,))
        p_XX.start()
        
        p_CC.join()
        p_XC.join()
        p_CX.join()
        p_XX.join()

        # Users in common
        # 
        # Finds which users are considered into the selection, without repetitions

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
                    
        print('\t Result size:', len(users_in_common))
        print('\t Result data:', users_in_common)

        label = []
        all_selected_users = []
        all_time_differences = []

        versions = ['CC', 'XC', 'CX', 'XX']
        temp_users = [list(selected_users_CC), list(selected_users_XC), list(selected_users_CX), list(selected_users_XX)]
        temp_td = [list(time_differences_CC), list(time_differences_XC), list(time_differences_CX), list(time_differences_XX)]
        temp_size = [len(selected_users_CC), len(selected_users_XC), len(selected_users_CX), len(selected_users_XX)]

        for l in range(0, 4):        
            z = temp_size.index(max(temp_size))
            label.append(versions[z])
            all_selected_users.append(temp_users[z])
            all_time_differences.append(temp_td[z])
            temp_size.pop(z) 
            versions.pop(z) 
            temp_users.pop(z)
            temp_td.pop(z)

        print(label)
        print(label[0], len(all_selected_users[0]), '||', len(selected_users_CC))
        print(label[1], len(all_selected_users[1]), '||', len(selected_users_XC))
        print(label[2], len(all_selected_users[2]), '||', len(selected_users_CX))
        print(label[3], len(all_selected_users[3]), '||', len(selected_users_XX))

        type_array = type_separation(all_selected_users, all_time_differences, tol)

        user_pairs, user_score_difference, fig = generate_pairs(type_array, df.Usuario, scores, trimming, label, plot=False)

        print('Plotting', len(user_pairs), 'pairs consisted of', len(users_in_common), 'users')
        #fig.set_size_inches(20, len(fig.axes) * 6)
        #plt.savefig(str(tol) + '/' + str(trimming) + '-distance.png')
        #print('Figure saved.')
        #plt.show()

        # ## Distance matrix

        matrix_size = len(users_in_common)

        distance_matrix = np.ones((matrix_size, matrix_size))

        for i in range(0, matrix_size):
            user_1 = users_in_common[i]
            for j in range(0, matrix_size):
                user_2 = users_in_common[j]
                for k in range(0, len(type_array)):
                    if (type_array[k][0][0] == user_1 and type_array[k][0][1] == user_2):
                        distance_matrix[i][j] = type_array[k][2]
                        distance_matrix[j][i] = type_array[k][2]
            clear()
            print( math.ceil(i*100/matrix_size), '% done (1)')
            
        distance_matrix = abs(distance_matrix)

        for i in range(0, matrix_size):
            for j in range(0, matrix_size):
                if (i == j):
                    distance_matrix[i][j] = 0
            clear()
            print( math.ceil(i*100/matrix_size), '% done (2)')
                    
        distance_matrix[np.isnan(distance_matrix)] = 1
                        
        #plt.figure(figsize=(15, 15))
        #ax = sns.heatmap(distance_matrix, annot=True, annot_kws={"size": 7}, fmt = '.2f')

        #ax.set(xticklabels=users_in_common, yticklabels=users_in_common)
        #plt.show()

        # ## Dendrogram

        plt.figure(figsize=(60, 28))  

        print(np.shape(distance_matrix))

        square = ssd.squareform(distance_matrix)

        linkage = sch.linkage(square, method='single')

        dend = sch.dendrogram(linkage, labels=users_in_common, leaf_rotation=90)

        plt.title("Dendrogram", size=40)  
        plt.xlabel('Users', size=40)
        plt.ylabel('Distance', size=40)
        plt.xticks(size = 20)
        plt.yticks(size = 40)

        plt.savefig(str(tol) + '/' + str(trimming) + '-dendrogram.png')

        #plt.show()

        type_array = type_separation(all_selected_users, all_time_differences, tol)

        type_array = np.array(type_array)
        user_score_difference = np.array(user_score_difference)

        total_exercises = type_array[:, 3]
        type_array = np.reshape(type_array[:, 2], (-1, 1))
        user_score_difference = np.reshape(user_score_difference, (-1, 1))

        analysing_data = np.concatenate((type_array, user_score_difference), axis=1)

        analysing_data = np.array(analysing_data, np.float)

        user_pairs_copy = user_pairs.copy()

        i = 0
        while i < len(analysing_data):
            if(np.isnan(analysing_data[i,1])):
                user_pairs_copy.pop(i)
                analysing_data = np.delete(analysing_data, i, 0)
                total_exercises = np.delete(total_exercises, i, 0)
            else:
                i = i + 1

        print("Finished")


        # ## Scatter plot

        plt.figure(figsize=(10, 7))
        plt.xlabel('Distance')
        plt.ylabel('Course final score difference between users')
        plt.title("Amount of exercises by user")

        scatter = plt.scatter(analysing_data[:, 0], analysing_data[:, 1], edgecolors = 'black', c=total_exercises, cmap='binary')

        plt.colorbar(scatter)

        #for i, txt in enumerate(user_pairs_copy):
        #    plt.annotate(txt, (analysing_data[i, 0], analysing_data[i, 1]))

        x = np.linspace(-1, 1, 201)
        y = [pow(i, 9) for i in x]
        plt.plot(x, y)

        axes = plt.gca()
        axes.set_xlim([-1.1, 1.1])
        axes.set_ylim([-1.1, 1.1])

        plt.grid(color='grey', linestyle='--', linewidth=.5)

        plt.savefig(str(tol) + '/' + str(trimming) + '-amount_of_exercises.png')

        #plt.show()


        # ## Distance from optimal curve (X^9)

        plt.figure(figsize=(10, 7))
        plt.xlabel('Distance')
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

        plt.savefig(str(tol) + '/' + str(trimming) + '-distance_from_curve.png')

        #plt.show()

        first = []
        second = []
        third = []
        fourth = []
        fifth = []
        others = []

        plt.figure(figsize=(10, 7))
        plt.xlabel('Distance')
        plt.ylabel('Course final score difference between users')
        plt.title("Hypotheses")

        for i in range(0, len(normal_points)):
            d = normal_points[i, 0]
            sc = normal_points[i, 1]
                    
            if (d >= 0.5 and sc >= 0.25):
                first.append([d, sc])
            elif (d >= 0.5 and sc > -0.25 and sc < 0.25):
                second.append([d, sc])
            elif (d > -0.5 and d < 0.5 and sc > -0.25 and sc < 0.25):
                third.append([d, sc])
            elif (d <= -0.5 and sc > -0.25 and sc < 0.25):
                fourth.append([d, sc])
            elif (d <= 0.5 and sc <= 0.25):
                fifth.append([d, sc])
            else:
                others.append([d, sc])
                
        first = np.asarray(first)
        second = np.asarray(second)
        third = np.asarray(third)
        fourth = np.asarray(fourth)
        fifth = np.asarray(fifth)
        others = np.asarray(others)
                
        plt.scatter(first[:, 0], first[:, 1], marker='o', color='blue', label='First', picker=True)
        plt.scatter(second[:, 0], second[:, 1], marker='o', color='green', label='Second', picker=True)
        plt.scatter(third[:, 0], third[:, 1], marker='o', color='black', label='Third', picker=True)
        plt.scatter(fourth[:, 0], fourth[:, 1], marker='o', color='pink', label='Fourth', picker=True)
        plt.scatter(fifth[:, 0], fifth[:, 1], marker='o', color='orange', label='Fifth', picker=True)

        x = np.linspace(-1, 1, 201)
        y = [pow(i, 9) for i in x]
        plt.plot(x, y)

        axes = plt.gca()
        axes.set_xlim([-1.1, 1.1])
        axes.set_ylim([-1.1, 1.1])
        axes.legend()

        plt.grid(color='grey', linestyle='--', linewidth=.5)

        plt.savefig(str(tol) + '/' + str(trimming) + '-hypotheses.png')

        #plt.show()


        # Checking outliers

        array_being_analysed = outliers

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
                    if ((user_1_ips[j] == user_2_ips[k]) == True):
                        amount = amount + min(count_1[j], count_2[k])
            #print('-'*100)
            #print('User 1 (', array_being_analysed[i][2], ',', len(user_1_ips), '): ', user_1_ips, count_1)
            #print('User 2 (', array_being_analysed[i][3], ',', len(user_2_ips), '): ', user_2_ips, count_2)
            #print('\nCorrelation: ', amount, np.intersect1d(user_1_ips, user_2_ips, assume_unique=True))
            ips_in_common.append(amount)

        plt.figure(figsize=(10, 7))
        plt.xlabel('Distance')
        plt.ylabel('Course final score difference between users')
        plt.title("Exercises with the same IP")

        scatter = plt.scatter(array_being_analysed[:, 0], array_being_analysed[:, 1], edgecolors = 'black', c=ips_in_common, cmap='binary')

        plt.colorbar(scatter)

        for i, txt in enumerate(ips_in_common):
            plt.annotate((txt, (array_being_analysed[i, 2], array_being_analysed[i, 3])), (array_being_analysed[i, 0], array_being_analysed[i, 1]))

        x = np.linspace(-1, 1, 201)
        y = [pow(i, 9) for i in x]
        plt.plot(x, y)

        axes = plt.gca()
        axes.set_xlim([-1.1, 1.1])
        axes.set_ylim([-1.1, 1.1])

        plt.grid(color='grey', linestyle='--', linewidth=.5)

        plt.savefig(str(tol) + '/' + str(trimming) + '-exercises_same_ip.png')

        #plt.show()

        # Material interaction index
        # By counting the number of events a user has, it can measure how many of them were directed towards reading the
        # materials. This is done by dividing the count of all the events by the count of the events where the user
        # tried to solve an exercise. However, since a user can try the same exercise multiple times, it was decided
        # that if the user tried the exercise at least one time (be it right or wrong), then it would count as an interaction
        # towards that exercise and that's it, no additional tries for that exercise would be considered as interactions.
        # Therefore, the formula is: number_of_exercises_tried / (number_of_material_interactions + number_of_exercises_tried).
        # The results mean: 1 -> purely exercise tryouts (fake account); 0,5 -> equal number of exercise tryouts and material reviews
        # (legit user) and; 0 -> purely material reviews (probably a professor or a material thief)
        

        interactions = check_interactions(array_being_analysed, df)

        plt.figure(figsize=(10, 7))
        plt.xlabel('Distance')
        plt.ylabel('Course final score difference between users')
        plt.title("Material interaction index")

        interactions_str = []
        
        for i in range(0, len(interactions)):
            interactions_str.append(str(str(round(interactions[i, 0], 2)) + ', ' + str(round(interactions[i, 1], 2))))
        
        scatter = plt.scatter(array_being_analysed[:, 0], array_being_analysed[:, 1], edgecolors = 'black', cmap='binary')

        for i, txt in enumerate(interactions_str):
            plt.annotate((txt, (array_being_analysed[i, 2], array_being_analysed[i, 3])), (array_being_analysed[i, 0], array_being_analysed[i, 1]))

        x = np.linspace(-1, 1, 201)
        y = [pow(i, 9) for i in x]
        plt.plot(x, y)

        axes = plt.gca()
        axes.set_xlim([-1.1, 1.1])
        axes.set_ylim([-1.1, 1.1])

        plt.grid(color='grey', linestyle='--', linewidth=.5)

        plt.savefig(str(tol) + '/' + str(trimming) + '-interaction_index.png')

        #plt.show()
        
        
    main()

