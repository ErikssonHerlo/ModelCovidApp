import streamlit as st
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


st.title("Regresión Lineal")
st.write("""
La regresión lineal es una técnica de modelado estadístico que se emplea para describir una variable de respuesta continua como una función de una o varias variables predictoras. Puede ayudar a comprender y predecir el comportamiento de sistemas complejos o a analizar datos experimentales, financieros y biológicos.
""")
st.write("""
Las técnicas de regresión lineal permiten crear un modelo lineal. Este modelo describe la relación entre una variable dependiente y (también conocida como la respuesta) como una función de una o varias variables independientes Xi (denominadas predictores).
""")
st.write("""
La ecuación general correspondiente a un modelo de regresión lineal simple es:
""")
st.latex("Y=β0+βiXi+ϵi")

st.subheader("Carga del Archivo")
st.write("""
Para realizar un análisis de regresión líneal, es necesario cargar un archivo de datos, con un formato específico. Estos pueden ser archivos con extensiones: csv, xls,xlsx o json.
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
        Elija las variables que se utilizarán para el análisis de regresión líneal """)
    st.markdown("#### Variable Independiente (X)")
    var_X = st.selectbox("Por favor elija una opción", df.keys(), key="variableX")
    st.markdown("#### Variable Dependiente (Y)")
    var_Y = st.selectbox("Por favor elija una opción", df.keys(), key="variableY")
    st.markdown("#### Valor de la Predicción")
    predValue = st.number_input("Ingrese el valor de la Predicción",None,None,0,1)
    st.markdown("#### Colores de la Gráfica")
    col1, col2 = st.columns(2)
    with col1:
        colorPoints = st.color_picker('Elije un Color para los Puntos de la gráfica','#EF280F')
    with col2:
        colorLine = st.color_picker('Elije un Color para la Tendencia de la gráfica', '#024A86')
    #Transformar Data a Array
    x = np.asarray(df[var_X]).reshape(-1, 1)
    # x = np.asarray(df[var_x])
    y = df[var_Y]

    # Regresion Lineal
    regr = LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    r2 = r2_score(y, y_pred)
    errorCuadratico = mean_squared_error(y, y_pred)

    predict = regr.predict([[predValue]])

    #Graficacion
    fig = plt.figure()
    plt.style.use("seaborn")
    plt.scatter(x, y, color= colorPoints)
    plt.plot(x, y_pred, color= colorLine)
    plt.title("Regresión Líneal")
    plt.ylabel(var_Y)
    plt.xlabel(var_X)
    #plt.savefig("linearRegression.png")
    #plt.close()

    #Obtenemos la imagen para mostrarla
    
    if st.button('Calcular'):
        st.subheader("Graficación")
        #image = Image.open("linearRegression.png")
        #st.image(image, caption = "Linear Regression")
        st.pyplot(fig)
        st.markdown("#### Datos de la Gráfica")

        col1, col2= st.columns(2)
        
        pendiente =  float(regr.coef_)
        indicadorPendiente = "+ Positiva" if pendiente>=0 else "- Negativa" 
        col1.metric("Pendiente", pendiente, indicadorPendiente)
        
        intercepto = float(regr.intercept_)
        indicadorIntercepto = "+ Positivo" if intercepto>=0 else "- Negativo"
        col2.metric("Intercepto", intercepto, indicadorIntercepto)
        
        col3, col4 = st.columns(2)
        col3.metric("Coeficiente de Determinación",r2)
        col4.metric("Error Cuadrático",errorCuadratico)
        
        st.subheader("Función de la Tendencia")
        
        operador = "+ " if intercepto>=0 else "" 
        st.latex(f"f(x)={pendiente}X {operador}{intercepto}")
        st.subheader("Predicción")
        indicadorPrediccion = "+ Positiva" if predict>=0 else "- Negativa"
        st.metric(f"El valor de la predicción para {predValue} es de: ",predict, indicadorPrediccion)




        
    


else:
    st.warning("Debe Cargar un Archivo Previamente")
   


st.sidebar.title("Indice")
st.sidebar.markdown("### [Carga del Archivo](#carga-del-archivo)")
st.sidebar.markdown("- [Contenido del Archivo](#contenido-del-archivo)")
st.sidebar.markdown("### [Parametrización](#parametrizaci-n)")
st.sidebar.markdown("- [Variable Indepentiente (X)](#variable-independiente-x)")
st.sidebar.markdown("- [Variable Depentiente (Y)](#variable-dependiente-y)")
st.sidebar.markdown("- [Valor de la Predicción](#valor-de-la-predicci-n)")
st.sidebar.markdown("- [Colores de la Gráfica](#colores-de-la-gr-fica)")
st.sidebar.markdown("### [Graficación](#graficaci-n)")
st.sidebar.markdown("- [Datos de la Gráfica](#datos-de-la-gr-fica)")
st.sidebar.markdown("### [Función de la Tendencia](#funci-n-de-la-tendencia)")
st.sidebar.markdown("### [Predicción](#predicci-n)")