# faster, but still slow
# fast subtractive convolutional algorithm? Nope. Not a convolution, as it would not need to get inverted
# Mathematical deduction on my scrapbook
def selectUsers(selected_users, tol, exercise_array_1, exercise_array_2, version, N_USERS, MIN_EXERCISES):
    for i in range(0, N_USERS - 1):
        for j in range(i + 1, N_USERS):
            time_dif = exercise_array_1[i] - exercise_array_2[j]
            if (version == 'XC'):
                print(time_dif)
                time_dif = time_dif[time_dif >= 0]
            elif (version == 'CX'):
                print(time_dif)
                time_dif = time_dif[time_dif <= 0]
            nbr_exercises = sum(abs(x) < tol for x in time_dif)
            if (nbr_exercises >= MIN_EXERCISES):
                print(version, 'added.', i, 'is user 1,', j, 'is user 2.', nbr_exercises, 'exercises.', tol, 'minutes tolerance')
                selected_users.append([i, j])
            
        if(i  == int(N_USERS / 4)):
            print(version, '25% done')
        elif(i  == int(N_USERS / 2)):
            print(version, '50% done')
        elif(i  == int(N_USERS / (4/3))):
            print(version, '75% done') 
    print(version, 'thread finished.', version, 'added', len(selected_users), 'users.')
    return