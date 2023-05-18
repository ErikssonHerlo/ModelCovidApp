import string
from tokenize import Number
from sklearn.neural_network import MLPClassifier
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

st.title("Redes Neuronales")
st.write("""
Las redes neuronales son modelos simples del funcionamiento del sistema nervioso. Las unidades básicas son las neuronas, que generalmente se organizan en capas.
""")
st.write("""
Una red neuronal es un modelo simplificado que emula el modo en que el cerebro humano procesa la información: Funciona simultaneando un número elevado de unidades de procesamiento interconectadas que parecen versiones abstractas de neuronas.
""")
st.write("""
Las unidades de procesamiento se organizan en capas. Hay tres partes normalmente en una red neuronal : una capa de entrada, con unidades que representan los campos de entrada; una o varias capas ocultas; y una capa de salida, con una unidad o unidades que representa el campo o los campos de destino. Las unidades se conectan con fuerzas de conexión variables (o ponderaciones). Los datos de entrada se presentan en la primera capa, y los valores se propagan desde cada neurona hasta cada neurona de la capa siguiente. al final, se envía un resultado desde la capa de salida.
""")

st.write("""
La red aprende examinando los registros individuales, generando una predicción para cada registro y realizando ajustes a las ponderaciones cuando realiza una predicción incorrecta. Este proceso se repite muchas veces y la red sigue mejorando sus predicciones hasta haber alcanzado uno o varios criterios de parada.
""")

st.write("""
Al principio, todas las ponderaciones son aleatorias y las respuestas que resultan de la red son, posiblemente, disparatadas. La red aprende a través del entrenamiento. Continuamente se presentan a la red ejemplos para los que se conoce el resultado, y las respuestas que proporciona se comparan con los resultados conocidos. La información procedente de esta comparación se pasa hacia atrás a través de la red, cambiando las ponderaciones gradualmente. A medida que progresa el entrenamiento, la red se va haciendo cada vez más precisa en la replicación de resultados conocidos. Una vez entrenada, la red se puede aplicar a casos futuros en los que se desconoce el resultado.
""")

st.subheader("Carga del Archivo")
st.write("""
Para realizar un análisis de redes neuronales, es necesario cargar un archivo de datos, con un formato específico. Estos pueden ser archivos con extensiones: csv, xls,xlsx o json.
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
        Elija las variables que se utilizarán para el análisis de las redes neuronales """)
    st.markdown("#### Variable Objetivo")
    var_Object = st.selectbox("Por favor elija una opción", df.keys(), key="variableObjetivo")

    
    size = len(df.keys()) -1
    st.markdown("#### Valor de las Capas")
    capasValue = st.text_input(f"Ingrese los valores de las 3 capas, seguidos de una coma", "10,10,10")
 
    st.markdown("#### Cantidad de Iteraciones")
    iteraciones = st.number_input("Ingrese el valor de la Predicción",1,1000,500,1)
 
    
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



    #Obtenemos la imagen para mostrarla
    
    if st.button('Calcular'):
        print(iteraciones)
        
        #Validacion de Capas vacias
        if(capasValue != ""):
            arrayCapas = capasValue.split(',')
            
            capas = list(map(int, arrayCapas))
            
            if(len(capas)==3):

                if(predValues != ""):
                    arrayValues = predValues.split(',')
                    #Label Encoder para Parametros de prediccion
                    arrayValues = le.fit_transform(arrayValues)

                    print(arrayValues)
                   
                    if(len(arrayValues) == size):
                        st.subheader("Visualización de las Tuplas")
                        st.dataframe(features)
                        st.subheader("Predicción")
                        #Entrenamos el modelo
                        mlp = MLPClassifier(hidden_layer_sizes=(capas),max_iter=iteraciones, alpha=0.0001,
                        solver="adam", random_state = 21, tol = 0.000000001)
                        mlp.fit(features,label)
                        #Prediccion del Modelo 
                        predict = mlp.predict([arrayValues])
                        #predictTransform = le.inverse_transform(predict)
                        
                        #predict = model.predict([[10, 10, 300, 0]])
                        st.metric(f"El valor de la predicción para los valores ingresados es de: ",str(predict), getSign(str(predict)))
                    else:
                        st.warning(f"Debe ingresar {size} valores para la predicción, separados por comas")
                else:
                    st.warning("Debe ingresar los valores para la predicción")
            else: 
                st.warning("La Cantidad de las Capas debe ser igual a 3")
        else:
            st.warning("Debe ingresar el valor de las 3 Capas")



        
    


else:
    st.warning("Debe Cargar un Archivo Previamente")
   


st.sidebar.title("Indice")
st.sidebar.markdown("### [Carga del Archivo](#carga-del-archivo)")
st.sidebar.markdown("- [Contenido del Archivo](#contenido-del-archivo)")
st.sidebar.markdown("### [Parametrización](#parametrizaci-n)")
st.sidebar.markdown("- [Variable Objetivo](#variable-objetivo)")
st.sidebar.markdown("- [Valor de las Capas](#valor-de-las-capas)")
st.sidebar.markdown("- [Cantidad de Iteraciones](#cantidad-de-iteraciones)")
st.sidebar.markdown("- [Valor de la Predicción](#valor-de-la-predicci-n)")
st.sidebar.markdown("### [Visualización de las Tuplas](#visualizaci-n-de-las-tuplas)")
st.sidebar.markdown("### [Predicción](#predicci-n)")






