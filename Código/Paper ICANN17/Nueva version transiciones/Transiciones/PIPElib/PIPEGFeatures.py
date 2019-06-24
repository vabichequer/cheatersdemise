#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:35:27 2018

@author: AngelPL MiguelAGGS

GENERAL FEATURES


"""

from PIPEAux import applyByUser, combineFeat

def filterListByEvents(listEvents, eventsFiltering):
    #Funcion que filtra los eventos es funcion de la lista eventos pasados por argumento
    # eventsFiltering son por los eventos que queremos filtrar
    # listEvents es la lista de eventos del usuario

    import pandas as pd
    import numpy
    
    # Comprobacion de si la lista esta vacia (si lo esta, daria error la funcion)
    if listEvents == []:
        return []
    #Pasa la lista de eventos a un dataframe
    df = pd.DataFrame(listEvents)
     
    #comprueba si los eventos del estudiantes estan en la lista de los eventos por los que se filtra    
    df = df.loc[df['evento'].isin(eventsFiltering)]
    
    indices = numpy.array(df.index.values)
    
    listaF = numpy.array(listEvents)
    
    return list(listaF[indices])

#NUEVO
def filterListByProblemID(listEvents, ID):
    # Filtra los problemas por eventos de problemas por id. listEvents es la
    #  lista de eventos filtrada por problem_check y ID se obtiene del dataframe
    #  de asignación de ejercicios/actividades/examen/aeJava...
    import pandas as pd
    import numpy
    
    # Comprobacion de si la lista esta vacia (si lo esta, daria error la funcion)
    if listEvents == []:
        return []
    #Pasa la lista de eventos a un dataframe
    df = pd.DataFrame(listEvents)
    
    #comprueba si los id naturales de los problemas estan en la lista.
    df = df.loc[df['id_natural'].isin(ID)]
    indices = numpy.array(df.index.values)
    listaF = numpy.array(listEvents)
    
    return list(listaF[indices])

def eventCounter(eventsDF):
    # Funcion que cuenta el numero de eventos y lo inclute en una columna de eventsDF

    # Contar eventos
    featuresDF = eventsDF.copy()
    featuresDF['NumEventos'] = featuresDF['Eventos'].apply(lambda x: len(x))
    del featuresDF['Eventos']
    
    return featuresDF


def countCorrectByList(listResProblem):
    #Funcion que cuenta el numero de problemas correctos en una lista
    import pandas as pd
    
    #A veces la lista de resultados puede venir vacía asique lo tomamos como 
    #  resultado no correcto (esto pasa muy pocas veces debe ser un error
    #  del sevidor de edX)
    if listResProblem == []:
        return 0;
    
    dfL = pd.DataFrame(listResProblem)
    # Contamos los problemas que son correctos
    countCorrecto = dfL[dfL["correcto"]=="True"]["correcto"].count()
    return countCorrecto
    
    
def correctProblemByList(listEvents):
    #Cuenta el numero de eventos correctos a partir de una lista de eventos   
    import pandas as pd
    
    #Si tenemos lista de eventos vacia contamos 0 problemas correctos
    if listEvents==[]:
        return 0
    
    #Transformamos la lista de eventos a dataframe
    df = pd.DataFrame(listEvents)

    #tenemos una lista de resultados por cada problema ya que estan formados por subejercicios,
    #por lo tanto tomamos la lista y la transformamos a dataframe en la funcion del apply
    df['numCorrectByUser'] = df['resultados'].apply(lambda x: countCorrectByList(x))
 
    return  df['numCorrectByUser'].sum()


def attemptsProblemByList(listEvents):
    # Cuenta el número de intentos en problemas.
    import pandas as pd
    
    #Si tenemos lista de eventos vacia contamos 0 intentos
    if listEvents==[]:
        return 0

    #Transformamos la lista de eventos a dataframe
    df = pd.DataFrame(listEvents)

    return  pd.to_numeric(df['num_intentos']).sum()


def sumPoints(listPartsPoints):
    #Funcion que suma los puntos de las partes de la autoevaluacion
    import pandas as pd
    
    #A veces la lista de resultados puede venir vacía asique lo tomamos como 0 puntos
    if listPartsPoints == []:
        return 0.0;
    
    dfL = pd.DataFrame(listPartsPoints)
    dfL[['puntos_autoevaluacion']] = dfL[['puntos_autoevaluacion']].apply(pd.to_numeric)
     
    return dfL["puntos_autoevaluacion"].sum()  
    
    
def sumPointsByList(listEvents):
    # Suma los puntos del proyecto
    import pandas as pd
    
    #Si tenemos lista de eventos vacia contamos que tiene un 0.0 en el proyecto
    if listEvents==[]:
        return 0.0
    
    #Transformamos la lista de eventos a dataframe
    df = pd.DataFrame(listEvents)

   
    df['self-evaluationPoints'] = df['partes'].apply(lambda x: sumPoints(x))
 
    return  df['self-evaluationPoints'].sum()

def countWordsByList(listEvents):
    # Cuenta el número de palabras del foro
    import pandas as pd
    
    #Si tenemos lista de eventos vacia contamos que tiene 0 palabras
    if listEvents==[]:
        return 0.0
    
    #Transformamos la lista de eventos a dataframe
    df = pd.DataFrame(listEvents)

    #Contamos el numero de palabras y lo guardamos en una nueva columna llamada Total_words
    df['Total_words'] = df['cuerpo'].apply(lambda x: len(x.split()))
 
    return  df['Total_words'].sum()

def GeneralFeatures(eventsDF, eventsWeek, assignationProblemCheckFile, verbose = 2):
    # Calcula las caracteristicas generales
    import pandas as pd
    import time
    
    # Num eventos x semana
    if verbose > 0:
        print("Number of events per week")   
    start = time.time()
    
    numEvents = {}
    for week in eventsWeek:
        # Contamos eventos
        numEvents[week] = eventCounter(eventsWeek[week])
        numEvents[week].rename(columns={'NumEventos': 'NumEventos_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
            
    # Num eventos video x semana
    if verbose > 0:
        print("Number of video events per week")
    start = time.time()
              
    numVideos = {}
    listEvents = ["play_video", "pause_video", "seek_video", "stop_video"]
    for week in eventsWeek:
        # Filtramos por evento filterListByEvents
        numVideos[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        numVideos[week] = eventCounter(numVideos[week])
        numVideos[week].rename(columns={'NumEventos': 'NumInteraccVideos_'+week}, inplace=True)
        
    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
        
              
    # Num eventos proyecto x semana
    if verbose > 0:
        print("Number of project events per week")
    start = time.time()
              
    numProyecto = {}
    listEvents = ["openassessmentblock.self_assess"]
    for week in eventsWeek:
        # Filtramos por evento
        numProyecto[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        numProyecto[week] = eventCounter(numProyecto[week])
        numProyecto[week].rename(columns={'NumEventos': 'NumInteraccProy_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # Num eventos documentos x semana
    if verbose > 0:
        print("Number of document events per week")
    start = time.time()
              
    numDocs = {}
    listEvents = ["textbook.pdf.chapter.navigated"]
    for week in eventsWeek:
        # Filtramos por evento
        numDocs[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        numDocs[week] = eventCounter(numDocs[week])
        numDocs[week].rename(columns={'NumEventos': 'NumInteraccDoc_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))

              
    # Num eventos foro x semana
    if verbose > 0:
        print("Number of forum events per week")
    start = time.time()
              
    numForo = {}
    listEvents = ["edx.forum.searched","edx.forum.comment.created", "edx.forum.response.created", "edx.forum.thread.created"]
    for week in eventsWeek:
        # Filtramos por evento
        numForo[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        numForo[week] = eventCounter(numForo[week])
        numForo[week].rename(columns={'NumEventos': 'NumInteraccForo_'+week}, inplace=True)
        
    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # Num eventos problemas (ejercicios + actividades) x semana
    if verbose > 0:
        print("Number of problem events per week")
    start = time.time()
              
    numProbs= {}
    listEvents = ["problem_check"]
    for week in eventsWeek:
        # Filtramos por evento
        numProbs[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        numProbs[week] = eventCounter(numProbs[week])
        numProbs[week].rename(columns={'NumEventos': 'NumInteraccProb_'+week}, inplace=True)
    
    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # NUEVO 
    # Leemos el fichero. Eliminamos IDs naturales replicados.
    assignationProblemCheck = pd.read_csv(assignationProblemCheckFile, index_col = 0)
    assignationProblemCheck = assignationProblemCheck.reset_index().drop_duplicates(subset=
                                            'id_natural', keep='last').set_index('id_natural')
    
    # NUEVO
    # Num eventos ejercicios x semana
    if verbose > 0:
        print("Number of video-exercise events per week")
    start = time.time()
              
    numEjs= {}
    listEvents = ["problem_check"]
    ProblemCheckFiltering = 'E'

    # Comprobamos si se han introducido elementos a descartar en problem check.
    APC = assignationProblemCheck
    ID = [str(i) for i in APC.index[(APC.tipo_evento == ProblemCheckFiltering)]]

    for week in eventsWeek:
        # Filtramos por evento
        numEjs[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        numEjs[week] = applyByUser(numEjs[week], function = filterListByProblemID, ID = ID)
        # Contamos eventos
        numEjs[week] = eventCounter(numEjs[week])
        numEjs[week].rename(columns={'NumEventos': 'NumInteraccEjer_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # NUEVO
    # Num eventos actividades x semana
    if verbose > 0:
        print("Number of activity events per week")
    start = time.time()
              
    numAcs= {}
    listEvents = ["problem_check"]
    ProblemCheckFiltering = 'A'

    # Comprobamos si se han introducido elementos a descartar en problem check.
    APC = assignationProblemCheck
    ID = [str(i) for i in APC.index[(APC.tipo_evento == ProblemCheckFiltering)]]

    for week in eventsWeek:
        # Filtramos por evento
        numAcs[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        numAcs[week] = applyByUser(numAcs[week], function = filterListByProblemID, ID = ID)
        # Contamos eventos
        numAcs[week] = eventCounter(numAcs[week])
        numAcs[week].rename(columns={'NumEventos': 'NumInteraccActiv_'+week}, inplace=True)
        
    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # Problemas correctos x semana
    if verbose > 0:
        print("Number of correct problems per week")
    start = time.time()
              
    correctProbs= {}
    listEvents = ["problem_check"]
    for week in eventsWeek:
        # Filtramos por evento
        correctProbs[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        correctProbs[week] = applyByUser(correctProbs[week], function = correctProblemByList)
        correctProbs[week].rename(columns={'Eventos': 'NumProbCorrect_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # Num intentos problemas x semana
    if verbose > 0:
        print("Number of problem attepts per week")
    start = time.time()
              
    attemptProbs= {}
    listEvents = ["problem_check"]
    for week in eventsWeek:
        # Filtramos por evento
        attemptProbs[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        attemptProbs[week] = applyByUser(attemptProbs[week], function = attemptsProblemByList)
        attemptProbs[week].rename(columns={'Eventos': 'NumProbAttepts_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # Nota proyecto x semana
    if verbose > 0:
        print("Project points per week")
    start = time.time()
              
    notaProy= {}
    listEvents = ["openassessmentblock.self_assess"]
    for week in eventsWeek:
        # Filtramos por evento
        notaProy[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        notaProy[week] = applyByUser(notaProy[week], function = sumPointsByList)
        notaProy[week].rename(columns={'Eventos': 'NotaProy_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # Palabras foro x semana
    if verbose > 0:
        print("Number of forum words per week")
    start = time.time()
              
    wordsForum= {}
    listEvents = ["edx.forum.comment.created", "edx.forum.response.created", "edx.forum.thread.created"]
    for week in eventsWeek:
        # Filtramos por evento
        wordsForum[week] = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
        # Contamos eventos
        wordsForum[week] = applyByUser(wordsForum[week] , function = countWordsByList)
        wordsForum[week].rename(columns={'Eventos': 'WordsForum_'+week}, inplace=True)

    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
              
    # NUEVA VERSION
    # aeJava correctos en curso
    if verbose > 0:
        print("Java self-evaluation correct in course")
    start = time.time()
              
    listEvents = ["problem_check"]
    ProblemCheckFiltering = 'J'

    # Comprobamos si se han introducido elementos a descartar en problem check.
    APC = assignationProblemCheck
    ID = [str(i) for i in APC.index[(APC.tipo_evento == ProblemCheckFiltering)]]

    # Filtramos por evento
    correctAEJ = applyByUser(eventsWeek[week], function = filterListByEvents, eventsFiltering = listEvents)
    correctAEJ = applyByUser(correctAEJ, function = filterListByProblemID, ID = ID)
    # Contamos eventos
    correctAEJ = applyByUser(correctAEJ, function = correctProblemByList)
    correctAEJ.rename(columns={'Eventos': 'NumSEJavaCorrect'}, inplace=True)
    
    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
              
    if verbose > 0:
        print('Finished!')
    
    # DF de caracteristicas semanales
    featWeek = {}

    for week in eventsWeek:
        featWeek[week] = combineFeat([numEvents[week], numVideos[week], numProyecto[week],
                                        numDocs[week], numForo[week], numProbs[week],
                                        numEjs[week], numAcs[week], correctProbs[week], 
                                        attemptProbs[week], notaProy[week], wordsForum[week]])

    # DF de características de curso
    featCourse = [correctAEJ]
    
    return featWeek, featCourse