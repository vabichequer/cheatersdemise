#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:50:24 2018

@author: AngelPL MiguelAGGS

AUXILIARES

applyByUser(eventsDF, function, Parallel = -1, **kwargs)
parallelizeApply(data, function, cores = 0, **kwargs)
wrapperApply(function, kwargs, data)

combineFeat(featList, dropNAN = True)
DF2CSV(dfToSave,filename)
CSV2DF(filename)
"""

def applyByUser(eventsDF, function, Parallel = -1, **kwargs):
    # Aplica una función por usuario (campo de Eventos). Puede ejecutarse en
    #  paralelo. 
    eventsDFProcessed = eventsDF.copy()
    eventsDFProcessed['Eventos'] = parallelizeApply(eventsDF['Eventos'], function, cores = Parallel, **kwargs)
    return eventsDFProcessed
    

def parallelizeApply(data, function, cores = 0, **kwargs):
    # Esta función toma como parametros de entrada un conjunto de datos,
    #  una función, el numero de cores y argumentos variables de entrada
    #  de esa funcion y ejecuta un apply de esa funcion en paralelo.
    from multiprocessing import cpu_count, Pool
    from functools import partial
    import numpy as np
    import pandas as pd

    # Comprobacion del numero de cores
    maxCores = cpu_count()
    if cores < 1 or cores > maxCores:
        cores = maxCores
    
    # Split de los datos
    data_split = np.array_split(data, cores)
    # Creacion de una pool
    pool = Pool(cores)
    # Llamamos al wrapper y mapeamos los resultados
    data = pd.concat(pool.map(partial(wrapperApply, function, kwargs), data_split))
    # Cerramos la pool y hacemos un join (wait)
    pool.close()
    pool.join()
    
    return data

# Funcion wrapper del apply
def wrapperApply(function, kwargs, data):
    data = data.apply(lambda x: function(x, **kwargs))
    return data


#EDITADO
#Combina las caracteristicas de la lista de dataframes pasadas por argumento
def combineFeat(featList, dropNAN = True, takeLast = False):
    import pandas as pd
    
    dfold = featList[0].copy()
    dfold.set_index('Usuario', inplace = True)
    
    for ind in range(1,len(featList)):
        dfnew = featList[ind].copy()
        dfnew.set_index('Usuario', inplace = True)
        dfold = pd.concat([dfold, dfnew], axis=1)

    dfold.reset_index(inplace = True)
    dfold.rename(columns = {'index':'Usuario'}, inplace = True)
    
    if dropNAN:
        dfold = dfold.dropna(axis = 0, how = 'any')
    else:
        dfold = dfold.fillna(0)
    
    if takeLast:
        dfold = dfold[:][(dfold.Usuario.isin(list(dfnew.index)))]
        
    return dfold

def cummulateFeatures(featWeek):
    
    dfCummulate = {}
    accumulation = []
    
    for week in sorted(featWeek):
        if accumulation == []:
            accumulation.append(featWeek[week])
            dfCummulate[week] = featWeek[week]
        else:
            accumulation.append(featWeek[week])
            dfCummulate[week] = combineFeat(accumulation, dropNAN = False, takeLast = True)
        
    return dfCummulate


def DF2CSV(dfToSave,filename):
    # Guarda dfToSave en el fichero filename
    dfToSave.to_csv(filename, sep='|', encoding='utf-8', index=False)

def CSV2DF(filename):
    import pandas as pd
    # Carga el fichero filename a un dataframe    
    csvDf = pd.read_csv(filename, sep='|')
    return csvDf

def dict2CSV(name,dictW):
    for week in dictW:
        filename= name+"_"+week+".csv"
        DF2CSV(dictW[week], filename)