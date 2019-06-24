#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:02:52 2018

@author: AngelPL MiguelAGGS

PROCESSING

readDataFile(filename)
dataFiltering(eventosDF, numEventos)
filterListByDate(listEvents, dates)
timeFiltering(eventsDF, dateDict = {}, Parallel = -1)
Preprocessing(filename, minTotalUserEvents = 0, dateDic = {}, minWeeklyUserEvents = 0, verbose = 2)
ready2Predict(dfToPredict, nameTarget)

"""

"""
### SEMANAS DE LA EDICION 1 ##

sem1 = datetime(2015,2,24)
sem2 = datetime(2015,3,3)
sem3 = datetime(2015,3,10)
sem4 = datetime(2015,3,17)
sem5 = datetime(2015,3,24)
sem6 = datetime(2015,3,31)
sem7 = datetime(2015,4,7)
sem8 = datetime(2015,4,14)


### SEMANAS DE EDICION 2 ##

sem1 = datetime(2015,10,5)
sem2 = datetime(2015,10,12)
sem3 = datetime(2015,10,19)
sem4 = datetime(2015,10,26)
sem5 = datetime(2015,11,2)
sem6 = datetime(2015,11,9)
sem7 = datetime(2015,11,16)
sem8 = datetime(2015,11,23)


dateDict = {"Week1": [sem1, sem2],
           "Week2": [sem2, sem3],
           "Week3": [sem3, sem4],
           "Week4": [sem4, sem5],
           "Week5": [sem5, sem6],
           "Week6": [sem6, sem7],
           "Week7": [sem7, sem8]}
"""

from PIPEAux import applyByUser

def conv(o):
    # Convierte los datetime a fecha en escritura
    from datetime import datetime
    if isinstance(o, datetime):
        return o.__str__()
    
def saveDateDict(dateDict, filename):
    # Guarda un diccionario de fechas
    import json
    with open(filename, 'w') as outfile:
        json.dump(dateDict, outfile, default = conv)
        
def loadDateDict(filename):
    # Carga un diccionario de fechas
    import json
    from datetime import datetime
    
    dateDict = json.load(open(filename, 'r'))
    for k in dateDict:
        for x in range(len(dateDict[k])):
            dateDict[k][x] = datetime.strptime(dateDict[k][x].split(" ")[0], '%Y-%m-%d')
    return dateDict

        
def readDataFile(filename):
    # Funcion que carga un fichero en un dataframe con el id del usuario
    #  en una columna y sus eventos en formato json en otra
    
    import json
    import pandas as pd
    
    # Leemos el fichero por lineas
    with open(filename) as f:
        users_events = f.readlines()
        
    # Parseamos a formato json cada linea
    listJson = list(map(json.loads, users_events));
   
    # Creamos el dataframe a partir de               
    df = pd.DataFrame.from_dict(listJson, orient='columns');
                               
    return df



def dataFiltering(eventosDF, numEventos):   
    #Funcion que filtra los usuarios en funcion de un numero de eventos pasado por argumento
    return eventosDF[eventosDF['Eventos'].apply(lambda x: len(x) > numEventos)]


def filterListByDate(listEvents, dates):
    #Funcion que filtra una lista de eventos en funcion de las fechas pasadas por argumento

    import numpy
    import pandas as pd

    # Transformamos la lista de eventos a dataframe
    df = pd.DataFrame(listEvents)
    # Convertimos el tiempo a datetime
    df['tiempo'] = pd.to_datetime(df['tiempo'])
    # Comparamos los datos que se encuentren entre las fechas indicadas
    df = df[(df['tiempo'] >= dates[0]) & (df['tiempo'] < dates[1])]
    # Tomamos los indices donde se cumple la condicion
    indices = numpy.array(df.index.values)
    # Tomamos la lista de eventos
    listaF = numpy.array(listEvents)
    # Devolvemos la lista de eventos en dichos indices
    return list(listaF[indices])


def timeFiltering(eventsDF, dateDict = {}, Parallel = -1):
    # Devuelve el diccionario de dataframes con los eventos por fechas
    
    # Inicializacion del diccionario de caracteristicas
    eventsDFSplit = {}
    
    # Si no hay diccionario, cuenta entre todas las fechas (diccionario por defecto)
    if dateDict == {}:
        dateDict = {"ALL": []}
    
    # Recorremos los distintos periodos de tiempo dados (ej: semanas) y extraemos las caracteristicas
    for periods in dateDict:
        eventsDFSplit[periods] = applyByUser(eventsDF, filterListByDate, 
                                             dates = dateDict[periods], Parallel = Parallel)
    
    return eventsDFSplit


def Preprocessing(filename, minTotalUserEvents = 0, dateDict = {}, minWeeklyUserEvents = 0, verbose = 2):
    # Carga, filtra por usuario, filtra pof fecha (y vuelve a filtrar por usuario) un fichero 
    #  con la información de MOOCs
    import time
    
    # Lectura del fichero
    start = time.time()
    eventsDFRaw = readDataFile(filename)
    eventsDFRaw = eventsDFRaw[eventsDFRaw['Usuario'] != '']
    end = time.time()
    
    
    # Print info
    if verbose > 0:
        print('Loaded users:', len(eventsDFRaw))
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))

    # Filtrado por usuario
    start = time.time()
    eventsDF = dataFiltering(eventsDFRaw, minTotalUserEvents);
    end = time.time()

    # Print info
    if verbose > 0:
        print('Users after filtering:', len(eventsDF))
    if verbose > 1:
        print(' - Time:', "%.2f [s]"%(end-start))
        
    # Filtrado por tiempo
    start = time.time()
    eventsWeek = timeFiltering(eventsDF, dateDict)
    end = time.time()

    # Print info
    if verbose > 1:
        print('Filtered by time')
        print(' - Time:', "%.2f [s]"%(end-start))

    # Filtrado por usuario en semana
    for week in eventsWeek:
        eventsWeek[week] = dataFiltering(eventsWeek[week], minWeeklyUserEvents)

    # Print info
    if verbose > 0:
        for week in eventsWeek:
            print(week, '- Users after filtering:', len(eventsWeek[week]))

    return eventsDF, eventsWeek


def ready2Predict(dfToPredict, nameTarget = ''):
    # Deja el dataframe preparado para llevarlo a prediccion: pone al principio
    # la columna de usuario, al final la de target y pasa a tipo entero aquellas
    #  variables sin decimales
    
    import math
    dfFinal={}
    for week in dfToPredict:

        # Ordenación primero columna de Usuario y final columna target
        columnas = dfToPredict[week].columns.tolist()
        if nameTarget != '':
            columnas.remove(nameTarget)
            columnas.append(nameTarget)
        columnas.remove("Usuario")
        columnas.insert(0,"Usuario")

        # Guardamos el nuevo dataframe
        dfFinal[week]=dfToPredict[week][columnas]
        
        # Recorremos las caracteristicas
        for col in dfFinal[week].columns:
            try:
                # Vemos si es entero y transformamos el tipo a entero
                isInt = (dfFinal[week][col].apply(lambda x: int(not(x - math.floor(x)) == 0)).sum() == 0)
                if isInt:
                    dfFinal[week][col] = dfFinal[week][col].astype(int)
            except:
                pass
            
    return dfFinal
