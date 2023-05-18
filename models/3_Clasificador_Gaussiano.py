import string
from tokenize import Number
import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, plot_tree

def getSign(valor: str) -> str:
    operador = "+ Positivo" if ('P' in valor) or ('p' in valor) or ('1' in valor)  else "- Negativo" 
    return operador

st.title("Clasificador Bayesiano de Gaussian")
st.write("""
Un clasificador de procesos gaussianos es un algoritmo de aprendizaje automático de clasificación.
 """)
st.write("""
Los procesos gaussianos son una generalización de la distribución de probabilidad gaussiana y se pueden utilizar como base para sofisticados algoritmos de aprendizaje automático no paramétricos para clasificación y regresión.
""")
st.write("""
Son un tipo de modelo de kernel, como las SVM, y a diferencia de las SVM, son capaces de predecir probabilidades de pertenencia a clases altamente calibradas, aunque la elección y configuración del kernel utilizado en el corazón del método puede ser un desafío.
""")
st.subheader("Procesos Gaussianos de Clasificación")

st.write("""
Las funciones de distribución de probabilidad gaussianas resumen la distribución de variables aleatorias, mientras que los procesos gaussianos resumen las propiedades de las funciones, por ejemplo, los parámetros de las funciones. Como tal, puede pensar en los procesos gaussianos como un nivel de abstracción o indirección por encima de las funciones gaussianas.
""")

st.subheader("Carga del Archivo")
st.write("""
Para realizar un análisis de clasificación gaussiana, es necesario cargar un archivo de datos, con un formato específico. Estos pueden ser archivos con extensiones: csv, xls,xlsx o json.
""")

uploadFile = st.file_uploader("Elija un archivo", type=['csv', 'xls', 'xlsx', 'json'])
if(uploadFile is not None):

    splitName = os.path.splitext(uploadFile.name)
    
    fileName = splitName[0]
    fileExtension = splitName[1]

    # Verificamos la extension del Archivo, para su lectura correspondiente
    if(fileExtension == ".csv"):
        df = pd.read_csv(uploadFile)
    elif(fileExtension == ".xls" or fileExtension == ".xlsx"):
        df = pd.read_excel(uploadFile)
    elif(fileExtension == ".json"):
        df = pd.read_json(uploadFile)

    # Imprimimos el contenido de la tabla
    st.markdown("#### Contenido del Archivo")
    st.dataframe(df)

    st.subheader("Parametrización")
    st.write("""
        Elija las variables que se utilizarán para el análisis del clasificador gaussiano """)
    st.markdown("#### Variable Objetivo")
    var_Object = st.selectbox("Por favor elija una opción", df.keys(), key="variableObjetivo")

    
    size = len(df.keys()) -1

    
    st.markdown("#### Valor de la Predicción")
    predValues = st.text_input(f"Ingrese los {size} valores de la predicción seguidos de una coma")
    #Transformar Data a Array
    field_y = df[var_Object].tolist()
    df = df.drop([var_Object], axis = 1)
    col_match = [s for s in df.head() if "NO" in s]
    if len(col_match) == 1: df = df.drop(['NO'], axis = 1)

    # Division de columnas
    fields_x = []
    le = preprocessing.LabelEncoder()
    headers = df.head()
    columns = headers.columns
    
    # Construccion de las tuplas
    for col in columns:
        col_list = df[col].tolist()
        #print(col_list)
        col_trans = le.fit_transform(col_list)
        #print(col_trans)
        fields_x.append(col_trans)
        #print("---------- ----------")
    print(fields_x)


    # Agregamos el arreglo de tuplas a la lista
    features = list(zip(*fields_x))
    label = le.fit_transform(field_y)
    #Entrenamos el modelo
    model = GaussianNB()
    model.fit(features, label)
   
   

    

    #Obtenemos la imagen para mostrarla
    
    if st.button('Calcular'):

        if(predValues != ""):
            arrayValues = predValues.split(',')
            #le2 = preprocessing.LabelEncoder()
            arrayValues = le.fit_transform(arrayValues)
              
            #print(arrayValues)
            #arrayIntValues = list(map(int, arrayValues))
            print("ARRAY VALUES")
            print(arrayValues)
            if(len(arrayValues) == size):
                st.subheader("Visualización de las Tuplas")
                st.dataframe(features)
                st.subheader("Predicción")
                #Prediccion del Modelo 

                predict = model.predict([arrayValues])
                #predictTransform = le.inverse_transform(predict)
                
                #predict = model.predict([[10, 10, 300, 0]])
                st.metric(f"El valor de la predicción para los valores ingresados es de: ",str(predict), getSign(str(predict)))
            else:
                st.warning(f"Debe ingresar {size} valores para la predicción, separados por comas")
        else:
            st.warning("Debe ingresar los valores para la predicción")



        
    


else:
    st.warning("Debe Cargar un Archivo Previamente")
   


st.sidebar.title("Indice")
st.sidebar.markdown("### [Carga del Archivo](#carga-del-archivo)")
st.sidebar.markdown("- [Contenido del Archivo](#contenido-del-archivo)")
st.sidebar.markdown("### [Parametrización](#parametrizaci-n)")
st.sidebar.markdown("- [Variable Objetivo](#variable-objetivo)")
st.sidebar.markdown("- [Valor de la Predicción](#valor-de-la-predicci-n)")
st.sidebar.markdown("### [Visualización de las Tuplas](#visualizaci-n-de-las-tuplas)")
st.sidebar.markdown("### [Predicción](#predicci-n)")


