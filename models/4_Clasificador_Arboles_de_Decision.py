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

st.title("Arboles de Decisión")
st.write("""
Un árbol de decisión es un modelo de predicción utilizado en diversos ámbitos que van desde la inteligencia artificial hasta la Economía. Dado un conjunto de datos se fabrican diagramas de construcciones lógicas, muy similares a los sistemas de predicción basados en reglas, que sirven para representar y categorizar una serie de condiciones que ocurren de forma sucesiva, para la resolución de un problema. 
""")
st.write("""
De forma gráfica, podemos definir un árbol de decisión, como un mapa de los posibles resultados de una serie de decisiones relacionadas. Permite que un individuo o una organización comparen posibles acciones entre sí según sus costos, probabilidades y beneficios. Se pueden usar para dirigir un intercambio de ideas informal o trazar un algoritmo que anticipe matemáticamente la mejor opción.
""")
st.write("""
Un árbol de decisión, por lo general, comienza con un único nodo y luego se ramifica en resultados posibles. Cada uno de esos resultados crea nodos adicionales, que se ramifican en otras posibilidades. Esto le da una forma similar a la de un árbol.
""")

st.write("""
Hay tres tipos diferentes de nodos: nodos de probabilidad, nodos de decisión y nodos terminales. Un nodo de probabilidad, representado con un círculo, muestra las probabilidades de ciertos resultados. Un nodo de decisión, representado con un cuadrado, muestra una decisión que se tomará, y un nodo terminal muestra el resultado definitivo de una ruta de decisión.
""")

st.subheader("Carga del Archivo")
st.write("""
Para realizar un análisis de un clasificación de arboles de decisión, es necesario cargar un archivo de datos, con un formato específico. Estos pueden ser archivos con extensiones: csv, xls,xlsx o json.
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
        Elija las variables que se utilizarán para el análisis del árbol de decisiones """)
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
    #print(fields_x)


    # Agregamos el arreglo de tuplas a la lista
    features = list(zip(*fields_x))
    label = le.fit_transform(field_y)

    #Entrenamos el modelo
    clf = DecisionTreeClassifier().fit(features,label)
    fig = plt.figure(figsize=(10,10))
    plt.style.use("seaborn")
    plot_tree(clf,filled = True)
    plt.title("Clasificador de Arboles de Decisión")

    # Errores


    #Prediccion
    
    #predict = model.predict(predValues)
    #predict = clf.predict([[10, 10, 300, 0]])

    #Obtenemos la imagen para mostrarla
    
    if st.button('Calcular'):
        if(predValues != ""):
            arrayValues = predValues.split(',')
            arrayValues = le.fit_transform(arrayValues)
            print(arrayValues)
            
            if(len(arrayValues) == size):
                st.subheader("Visualización de las Tuplas")
                st.dataframe(features)
                st.subheader("Graficación")
                st.pyplot(fig)
                st.subheader("Predicción")
                #Prediccion del Modelo 
                predict = clf.predict([arrayValues])
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
st.sidebar.markdown("### [Graficación](#graficaci-n)")
st.sidebar.markdown("### [Predicción](#predicci-n)")



