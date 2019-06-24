#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 20:10:22 2018

@author: AngelPL

Target
"""
from PIPEAux import applyByUser, combineFeat, CSV2DF
from PIPEGFeatures import filterListByEvents, eventCounter
from PIPEProcessing import filterListByDate

def hasNotPerformedAnyEvent(eventsDF, dates, kind = []):
    # Devuelve si el estudiante no ha realizado algun evento en un rango de fechas.
    # Si kind es [] valora cualquier tipo de evento, si no, los tipos de eventos pasados
    # en dicha lista.
    
    # Filtramos por rango de fechas
    eventsDFFiltered = applyByUser(eventsDF, function = filterListByDate, dates = dates)

    # Si kind esta lleno, filtramos por la lista de eventos
    if kind != []:
        eventsDFFiltered = applyByUser(eventsDFFiltered, function = filterListByEvents, eventsFiltering = kind)

    # Contamos el numero de eventos    
    returnDF = eventCounter(eventsDFFiltered)
    
    # Si no ha hecho eventos, ponemos a True
    returnDF["dropout"] = (returnDF["NumEventos"] < 1).map(int)
        
    del returnDF['NumEventos']
    
    return returnDF



def dropoutCrit1(eventsDF, dateDict = {}):
    #devuleve el target de abandono a 1 si el usuario no ha hecho actividades 
    # evaluables en las proximas dos semanas
    
    # Inicializacion del diccionario de caracteristicas target
    targetDict = {}
    
    # Definimos los eventos evaluables
    evaluable = ['openassessmentblock.self_assess', 'problem_check']

    # Ordenamos el diccionario, en el caso en el que no estuviese ordenado
    import operator
    sortedDict = sorted(dateDict.items(), key=operator.itemgetter(1))
    
    # Recorremos los distintos periodos de tiempo dados (ej: semanas) y extraemos las caracteristicas target
    for ind in range(len(sortedDict)-2): # -2 porque no podemos predecir las dos ultimas semana
        
        # Vamos de la fecha de inicio de la semana siguiente a la fecha del final de dentro de dos semanas
        dates = [sortedDict[ind+1][1][0], sortedDict[ind+2][1][1]]
        
        # Comprobamos si no ha realizado ningun evento evaluable en ese rango de tiempo
        targetDict[sortedDict[ind][0]] = hasNotPerformedAnyEvent(eventsDF, dates, evaluable)
        targetDict[sortedDict[ind][0]].rename(columns = {'dropout':'dropout1'})
        
    return targetDict



def dropoutCrit2(eventsDF, dateDict = {}):
    #devuelve el target de abandono a 1 si el usuario no ha hecho ningun evento la proxima semana

    # Inicializacion del diccionario de caracteristicas target
    targetDict = {}
    
    # Ordenamos el diccionario, en el caso en el que no estuviese ordenado
    import operator
    sortedDict = sorted(dateDict.items(), key=operator.itemgetter(1))
   
    # Recorremos los distintos periodos de tiempo dados (ej: semanas) y extraemos las caracteristicas target
    for ind in range(len(sortedDict)-1): # -1 porque no podemos predecir la ultima semana
        
        # Vamos de la fecha de inicio y fin de la semana siguiente
        dates = sortedDict[ind+1][1]
        
        # Comprobamos si no ha realizado ningun evento en ese rango de tiempo
        targetDict[sortedDict[ind][0]] = hasNotPerformedAnyEvent(eventsDF, dates)
        targetDict[sortedDict[ind][0]].rename(columns = {'dropout':'dropout2'})

        
    return targetDict


def clusterSospechoso(fileID):
    # Carga los IDs de usuarios del cluster sospechoso y les da valor 1
    import pandas as pd
    
    target = pd.read_csv(fileID, index_col = 0)
    target.Usuario = target.Usuario.apply(str)

    target['suspect'] = [1]*len(target)
    
    return target


def computeTarget(target, events, arg):
    # Calcúla diferentes tipos de target
    # Criterio de dropout 1 toma eventos total y dic de semana
    if target == 'dropout1':
        assert type(events) != dict
        assert type(arg) == dict
        outputDF = dropoutCrit1(events, arg)
        
    # Criterio de dropout 2 toma eventos total y dic de semana
    elif target == 'dropout2':
        assert type(events) != dict
        assert type(arg) == dict
        outputDF = dropoutCrit2(events, arg)
        
    # Nota toma dict de dataframes y fichero de nota
    elif target == 'nota':
        assert type(events) == dict
        assert type(arg) == str
        
        nota = CSV2DF(arg)
        nota.Usuario = nota.Usuario.apply(str)
        outputDF = {}
        for week in events:
            outputDF[week] = nota
            
    # Cluster sospechoso toma dict de dataframes y fichero de cluster sospechoso
    elif target == 'suspect':
        assert type(events) == dict
        assert type(arg) == str
        
        suspect = clusterSospechoso(arg)
        outputDF = {}
        
        for week in events:
            outputDF[week] = combineFeat([events[week], 
                        suspect], dropNAN = False)[['Usuario', 'suspect']]
        
    else:
        print("Error! Target not defined!")
        outputDF = {}
        
    return outputDF
        