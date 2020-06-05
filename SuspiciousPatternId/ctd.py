# As the time differences are calculated, when a nan comes up, the result is always nan. Therefore, if a user
# did the exercise and the other don't, no matter the time the one who did has, it will be a nan, because
# the one who didn't do the exercise will have a nan.
def computeTimeDifferences(time_difference, selected_users, exercise_array_1, exercise_array_2, version, N_EXERCISES):
    for i in range(0, len(selected_users)):
        td_temp = []
        j = selected_users[i][0]
        k = selected_users[i][1]
        
        #print("Users:", j, "vs", k)

        for l in range(0, N_EXERCISES):
            total = exercise_array_1[j][l] - exercise_array_2[k][l]
            td_temp.append(total)
            #if (total <= 2):
                #print(l)


        time_difference.append(td_temp)
    print(version, 'finished.')
    return
