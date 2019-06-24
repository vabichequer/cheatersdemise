#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 13:41:04 2018

@author: AngelPL

TRANSICIONES

# Ejemplo de dicEvents y param

dicEvents = {
    'V' : ['play_video', 'seek_video'],
    'N' : ['problem_check'],
    'F' : ['edx.forum.searched', 'edx.forum.comment.created', 
           'edx.forum.response.created', 'edx.forum.thread.created'],
    'P' : ['openassessmentblock.self_assess'],
    'D' : ['textbook.pdf.chapter.navigated']
}

param = {'dicEvents': dicEvents,
        'assignationProblemCheckFile': 'IDNaturalProblemas.csv',
        'ProblemCheckExceptions': ['J','X'],
        'pausaMin': 60,
        'useFreq': True,
        'deleteZeroTrans': False}


event2eventType(event, dicEventsReverse, assignationProblemCheck)
addInactivity(dfUserEventType, sleepTimeMin)
videoFiltering(dfUserEventType)
computeUserTransitions(dfUserEventType)
userTransitions(listEvents, dicEventsReverse, assignationProblemCheck, videoFilt = True, 
                    pausaMin = 0, transicionesEspecificas = False)
Transitions(eventsDF, param, Parallel = -1)
invDict(dic)

"""
from PIPEAux import parallelizeApply

def TransitionFeatures(eventsWeek, param, verbose = 2):
    import time
    
    # Num eventos foro x semana
    if verbose > 0:
        print("Transitions per week")
    start = time.time()
              
    transitions = {}
    for week in eventsWeek:
        # Filtramos por evento
        transitions[week] = Transitions(eventsWeek[week], param)
        transitions[week].columns = [name+"_"+week if name != 'Usuario' else name for name in transitions[week].columns]
        
    end = time.time()
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))

    return transitions

def event2eventType(event, dicEventsReverse, assignationProblemCheck):
    # Transforma un evento a un "tipo de evento".
    # Elimina eventos no considerados. Distingue entre tipos de problem_check.
    # Devuelve tipo de evento, id del evento y la marca temporal.

    import pandas as pd
    
    # Valores iniciales
    eventType = ''
    ID = ''
    tiempo = ''
    
    # Eliminacion de valores perdidos del dataframe de eventos de usuario.
    # Ejemplo: un problem_check tiene id_documento a NaN.
    event.dropna(inplace = True)
    
    # Si el eventType no se encuentra en el diccionario, se considera un
    #  eventType nulo o no considerado. Ejemplo: load_video.
    try:
        eventType = dicEventsReverse[str(event['evento'])]
    except:
        eventType = '' # BRICK NULO, NO CONSIDERADO
    
    # Asignacion eventType, ID y timestamp
    # Caso "problem_check"
    if eventType == 'N':
        # ID natural y timestamp
        ID = str(event['id_natural'])
        tiempo = pd.to_datetime(event['tiempo'])
        
        # Si no esta en la lista de asignacion, no consideramos el evento.
        # Ejemplo: cuando no se consideran los ejercicios de Java (J) o de examen (X).
        try:
            eventType = assignationProblemCheck['tipo_evento'][int(ID)]
        except:
            eventType = ''
            
    # Cualquier otro caso no "problem_check"
    elif eventType != '':
        # timestamp
        tiempo = pd.to_datetime(event['tiempo'])
        
        # algunos eventos no tienen ID. Ejemplo: busqueda en foro.
        try:
            # Esta funcion busca un campo que comience por "id_".
            # Ejemplo: id_documento, id_foro...
            ID = event[[i for i in event.index if i.startswith('id_')][0]]
        except:
            ID = ''
    
    # df de retorno. Si el eventType es considerado, lo devolvemos. Si no, devolvemos NaN
    if eventType != '':
        dfReturn = pd.Series([eventType, ID, tiempo], ['eventType', 'id', 'tiempo'])
    else:
        dfReturn = pd.Series([float('nan'), float('nan'), float('nan')], ['eventType', 'id', 'tiempo'])
    
    return dfReturn


def addInactivity(dfUserEventType, sleepTimeMin):
    # Anade un tiempo de inactividad entre tipos de evento cuando pase mas de 
    #  sleepTimeMin minutos entre eventos.
    
    import datetime as dt
    import pandas as pd
    
    # Tiempo de pausa en minutos
    sleepTime = dt.timedelta(minutes = sleepTimeMin)

    # Tiempo inicial (del primer evento)
    tiempo1 = [x.to_pydatetime() for x in dfUserEventType.tiempo]

    # Tiempo final (del siguiente evento)
    tiempo2 = tiempo1[:]
    tiempo2.append(tiempo2[len(tiempo2)-1] + sleepTime + dt.timedelta(seconds = 1))
    tiempo2.pop(0)

    # Intervalo que genera una inactividad
    sleep = [['S', sleepTimeMin, pd.to_datetime(tiempo1[x] + sleepTime)] 
             for x in range(len(tiempo1)) if tiempo2[x]-tiempo1[x] > sleepTime]
    
    # dataframe de tiempo de inactividad
    dfSleep = pd.DataFrame(sleep, columns = dfUserEventType.columns)

    # concatenar con el df de eventos
    dfUserEventType = pd.concat([dfUserEventType, dfSleep])
    
    # ordenarlos por timestammp
    dfUserEventType.sort_values(['tiempo'], inplace = True)
    dfUserEventType.reset_index(inplace = True, drop = True)
    
    return dfUserEventType


def videoFiltering(dfUserEventType):
    # Elimina eventos sobre el mismo video que esten consecutivos
    
    # Inicializacion
    indices = []
    id_video = ''
    
    # For sobre todos los eventos del usuario
    for x in dfUserEventType.index:
        
        # si el evento no es video, se resetea el id
        if dfUserEventType.loc[x,'eventType'] != 'V':
            id_video = ''
            
        # si es video y tiene el mismo id que el anterior,
        #  se anade a la lista de indices a eliminar
        elif id_video == dfUserEventType.loc[x,'id']:
            indices.append(x)
            
        # si es otro video, inicializamos nuevo id
        else:
            id_video = dfUserEventType.loc[x,'id']
        
    # Borramos las entradas de videos consecutivos
    dfUserEventType = dfUserEventType.drop(indices)
    return dfUserEventType


def computeUserTransitions(dfUserEventType):
    import numpy as np
    import pandas as pd

    # lista de eventos con nombre unico
    eventType = np.unique(dfUserEventType.eventType)
    
    # generamos un diccionario cuya clave es el eventType y valor un indice empezando por 0
    events = {eventType[x]: x for x in range(len(eventType))}

    # Creacion de la matriz de transiciones
    M = np.zeros((len(eventType),len(eventType)))

    # Recorremos todos los eventos por usuario hasta el total -1
    indices = dfUserEventType.index
    for ind in range(len(indices)-1):
        # Obtenemos el indice origen y destino y anadimos un uno a la matriz
        #  donde corresponda
        orig = events[dfUserEventType.loc[indices[ind], 'eventType']]
        dest = events[dfUserEventType.loc[indices[ind+1], 'eventType']]
        M[orig,dest] += 1

    # Creacion de los nombres de las caracteristicas
    names = [x+'->'+y for x in eventType for y in eventType]
    
    # Matriz a vector de caracteristicas
    M = M.flatten()

    # Devolvemos las caracteristicas por usuario
    dfTrans = pd.Series(M.astype('int'),index=names)
    return dfTrans


def userTransitions(listEvents, dicEventsReverse, assignationProblemCheck, videoFilt = True, 
                    pausaMin = 0, transicionesEspecificas = False):
    # Calcula las transiciones por usuario a partir de la lista de eventos por usuario.
    # Para ello, primero genera una lista de "tipos de eventos" considerados,
    #  (las transiciones pueden ser especificas). Se puede incluir tiempo de inactividad,
    #  eliminacion de videos con el mismo id consecutivos.
    
    import pandas as pd
    
    # Transforma la lista de eventos a dataframe
    dfUserEvents = pd.DataFrame(listEvents)
    
    # Aplica la funcion event2eventType (eventType, ID, tiempo)
    dfUserEventType = dfUserEvents.apply(lambda x: event2eventType(x, 
                            dicEventsReverse, assignationProblemCheck), axis = 1)
    
    # Elimina los tipos de evento no considerados
    dfUserEventType.dropna(inplace = True)

    # Si el dataframe esta vacio, salimos de la funcion.
    # Esto se hace aqui porque puede que tengamos eventos que no consideramos.
    if dfUserEventType.empty:
        return pd.Series()
    
    # Las transiciones especificas tienen como tipo de evento el eventType_id
    if transicionesEspecificas:
        dfUserEventType['eventType'] = dfUserEventType['eventType'] + '_' + dfUserEventType['id']
        
    # Calculo del tiempo de inactividad (anadir pausa 'S')
    if pausaMin > 0:
        dfUserEventType = addInactivity(dfUserEventType, pausaMin)
        
    # Eliminamos videos consecutivos con mismo ID
    if videoFilt:
        dfUserEventType = videoFiltering(dfUserEventType)
        
    # Calculamos las transiciones por usuario
    dfUserEventType = computeUserTransitions(dfUserEventType)
    
    return dfUserEventType


def Transitions(eventsDF, param, Parallel = -1):
    import pandas as pd
    
    featuresDF = eventsDF.copy()
    
    # Comprobamos si se ha introducido un diccionario de eventos a considerar
    try:
        dicEvents = param['dicEvents']
    except:
        print('Error! Diccionario de eventos necesario (dicEvents).')
        return
        
    # Invertimos el diccionario (valor es ahora clave)
    def invDict(dic):
        return {vi: k for k, v in dic.items() for vi in v}
    dicEventsReverse = invDict(dicEvents)

    # Comprobamos si se ha introducido un fichero con los IDs naturales
    try:
        assignationProblemCheckFile = param['assignationProblemCheckFile']
    except:
        print('Error! Documento con ID natural de problemas necesario (assignationProblemCheckFile).')
        return
    
    # Leemos el fichero. Eliminamos IDs naturales replicados.
    assignationProblemCheck = pd.read_csv(assignationProblemCheckFile, index_col = 0)
    assignationProblemCheck = assignationProblemCheck.reset_index().drop_duplicates(subset=
                                            'id_natural', keep='last').set_index('id_natural')
    
    # Comprobamos si se han introducido elementos a descartar en problem check.
    # Si es asi, eliminamos estos elementos de la asignacion.
    try:
        ProblemCheckExceptions = param['ProblemCheckExceptions']
        for PCE in ProblemCheckExceptions:
            indicesDrop = assignationProblemCheck.index[(assignationProblemCheck.tipo_evento == PCE)]
            assignationProblemCheck.drop(indicesDrop, inplace = True)
    except:
        pass
    
    # Comprobamos si se desea filtrado de video
    try:
        videoFilt = param['videoFilt']
    except:
        videoFilt = True
    
    # Comprobamos si se desean anadir informacion de inactividad
    try:
        pausaMin = param['pausaMin']
    except:
        pausaMin = 0
    
    # Comprobamos si se desean computar transiciones especificas
    try:
        transicionesEspecificas = param['transicionesEspecificas']
    except:
        transicionesEspecificas = False
    
    # Comprobamos si se desea normalizar las transiciones por usuario
    try:
        useFreq = param['useFreq']
    except:
        useFreq = False
        
    # Comprobamos si se desea eliminar columnas no informativas
    try:
        deleteZeroTrans = param['deleteZeroTrans']
    except:
        deleteZeroTrans = False
    
    # Calculo de transiciones
    featuresDF = parallelizeApply(featuresDF.Eventos, userTransitions, cores = Parallel, dicEventsReverse =
                                  dicEventsReverse, assignationProblemCheck = assignationProblemCheck, videoFilt = videoFilt,
                                  pausaMin = pausaMin, transicionesEspecificas = transicionesEspecificas)    
    # Las transiciones nan son cero
    featuresDF = featuresDF.fillna(0).astype(int)
    
    # Eliminacion de conjuntos de transiciones nulas por todos los usuarios
    if deleteZeroTrans:
        featuresDF = featuresDF.loc[:, (featuresDF != 0).any(axis=0)]
    
    # Si se desea normalizar, normalizamos las transiciones por usuario
    if useFreq:
        featuresDF = featuresDF.div(featuresDF.sum(axis=1), axis=0)
        featuresDF = featuresDF.fillna(0.)
    
    # Añadimos la columna de identificacion de Usuario
    featuresDF = pd.concat([eventsDF['Usuario'], featuresDF], axis = 1)
    
    return featuresDF
