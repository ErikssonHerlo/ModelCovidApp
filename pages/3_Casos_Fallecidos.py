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
from datetime import datetime, timedelta, date

def getOperator(valor: float) -> str:
    operador = "+ " if valor>=0 else "" 
    return operador

st.title("Casos Fallecidos")
st.write("""
Se le llama así a una persona fallecida cumpliendo con la definición de caso confirmado; los casos fallecidos se visualizan por fecha de defunción. 
Es importante indicar que los casos suelen reportarse días después de su defunción por el tiempo de investigación individualizado y el proceso de notificación 
""")
st.write("Donde: ")
st.write("- **Fallecidos por Fecha:** representa el número total de Fallecimientos que se reportaron ese día y fueron informados a las autoridades.")
st.divider()
st.subheader("Carga del Archivo")
st.write("""
Para realizar un análisis de regresión polinomial, es necesario cargar un archivo de datos, con un formato específico. Estos pueden ser archivos con extensiones: csv, xls,xlsx o json.
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
    #Get Initial Date from Data
    initialDateString = df['fecha'][0]
    initialDate = datetime.strptime(initialDateString, '%Y-%m-%d').date()
    
    finalDateString = df['fecha'].iloc[-1]
    finalDate = datetime.strptime(finalDateString, '%Y-%m-%d').date()


    st.subheader("Parametrización")
    st.write("""
        Elija las variables que se utilizarán para el análisis de regresión líneal """)
    st.markdown("#### Variable Independiente (X)")
    var_X = st.selectbox("Por favor elija una opción", df.keys(), key="variableX")
    st.markdown("#### Variable Dependiente (Y)")
    var_Y = st.selectbox("Por favor elija una opción", df.keys(), key="variableY")
    column1, column2 = st.columns(2)
    with column1:
        st.markdown("#### Grado de la Función")
        grado = st.slider("Elija el Grado de la Función", 2, 5, 2, 1)
    st.markdown("#### Valor de la Predicción")
    datePrediction = st.date_input( "Ingrese la fecha que desea predecir", date.today(), min_value=initialDate)
    
    st.markdown("#### Inicio de la Muestra")
    initialSampleDate = st.date_input( "Ingrese la Fecha de Inicio de la Muestra", initialDate, min_value=initialDate, max_value=finalDate)
      
    totalDays = (finalDate - initialDate).days
    daysToPredict = (datePrediction - initialDate).days
    predValue = (datePrediction - initialSampleDate).days
    valueSample = (finalDate - initialSampleDate).days
    valueMinSample = totalDays - predValue
    st.write("- Dias desde el Inicio de la Pandemia hasta la fecha a predecir: ", daysToPredict)
    st.write("- Dias desde el Inicio de la Muestra hasta la fecha a predecir: ", predValue)
    st.markdown("#### Colores de la Gráfica")
    col1, col2 = st.columns(2)
    with col1:
        colorPoints = st.color_picker('Elije un Color para los Puntos de la gráfica','#EF280F')
    with col2:
        colorLine = st.color_picker('Elije un Color para la Tendencia de la gráfica', '#024A86')


    #Filtrado de Muestra
    df_filteredBySample = df.tail(valueSample)
    st.dataframe(df_filteredBySample)
    df_complete = df
    df = df_filteredBySample
    
    #Transformar Data a Array
    x_complete = np.asarray(df_complete[var_X]).reshape(-1, 1)
    y_complete = df_complete[var_Y]

    x = np.asarray(df[var_X]).reshape(-1, 1)
    y = df[var_Y]

    

    # Regrsion Polinomial
    pf = PolynomialFeatures(degree = grado)
    x_trans = pf.fit_transform(x)
    x_trans_complete = pf.fit_transform(x_complete)

    # Regresion Lineal
    regr = LinearRegression()
    regr.fit(x_trans, y)
    
    regr2 = LinearRegression()
    regr2.fit(x_trans_complete, y_complete)
    

    # Errores
    y_pred = regr.predict(x_trans)
    y_pred_complete = regr2.predict(x_trans_complete)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    errorCuadratico = mean_squared_error(y, y_pred)

    #Prediccion
    x_new_min = daysToPredict
    x_new_max = daysToPredict
    x_new = np.linspace(x_new_min, x_new_max, 1)
    x_new = x_new[:, np.newaxis]
    x_trans = pf.fit_transform(x_new)
    predict = regr.predict(x_trans)


    #Graficacion Completa
    fig = plt.figure()
    plt.style.use("seaborn")
    plt.scatter(x_complete, y_complete, color= colorPoints)
    plt.plot(x_complete, y_pred_complete, color= colorLine)
    plt.title(f"Regresión Polinomial de Grado {grado}")
    plt.ylabel(var_Y)
    plt.xlabel(var_X)

    #Graficacion por Intervalo
    fig_Complete = plt.figure()
    plt.style.use("seaborn")
    plt.scatter(x, y, color= colorPoints)
    plt.plot(x, y_pred, color= colorLine)
    plt.title(f"Regresión Polinomial de Grado {grado}")
    plt.ylabel(var_Y)
    plt.xlabel(var_X)
    #Obtenemos la imagen para mostrarla
    
    if st.button('Calcular'):
        st.subheader("Graficación")
        #image = Image.open("linearRegression.png")
        #st.image(image, caption = "Linear Regression")
        st.pyplot(fig)
        st.pyplot(fig_Complete)
        st.markdown("#### Datos de la Gráfica")

        col1, col2= st.columns(2)
        col1.write("Coeficientes de la Función")
        col1.write(regr.coef_)
        # pendiente =  float(regr.coef_)
        #indicadorPendiente = "+ Positiva" if pendiente>=0 else "- Negativa" 
        #col1.metric("Coeficientes de la Función", regr.coef_)
        
        intercepto = float(regr.intercept_)
        indicadorIntercepto = "+ Positivo" if intercepto>=0 else "- Negativo"
        col2.metric("Intercepto", intercepto, indicadorIntercepto)
        
        col3, col4 = st.columns(2)
        col3.metric("Coeficiente de Determinación",r2)
        col4.metric("Error Cuadrático Medio", rmse)
        
        st.subheader("Función de la Tendencia")
        
        if(grado == 2):
            b1 = float(regr.coef_[1])
            b2 = float(regr.coef_[2])
            
            st.latex(f"f(x)={b2}X^2 {getOperator(b1)} {b1}X {getOperator(intercepto)}{intercepto}")
        elif(grado == 3):
            b1 = float(regr.coef_[1])
            b2 = float(regr.coef_[2])
            b3 = float(regr.coef_[3])
            
            st.latex(f"f(x)= {b3}X^3 {getOperator(b2)}{b2}X^2 {getOperator(b1)} {b1}X {getOperator(intercepto)}{intercepto}")

        elif(grado == 4):
            b1 = float(regr.coef_[1])
            b2 = float(regr.coef_[2])
            b3 = float(regr.coef_[3])
            b4 = float(regr.coef_[4])
            
            st.latex(f"f(x)= {b4}X^4 {getOperator(b3)}{b3}X^3 {getOperator(b2)}{b2}X^2")
            st.latex(f"{getOperator(b1)} {b1}X {getOperator(intercepto)}{intercepto}")

        elif(grado == 5):
            b1 = float(regr.coef_[1])
            b2 = float(regr.coef_[2])
            b3 = float(regr.coef_[3])
            b4 = float(regr.coef_[4])
            b5 = float(regr.coef_[5])
            
            st.latex(f"f(x)= {b5}X^5 {getOperator(b4)}{b4}X^4 {getOperator(b3)}{b3}X^3")
            st.latex(f"{getOperator(b2)}{b2}X^2 {getOperator(b1)} {b1}X {getOperator(intercepto)}{intercepto}")
        #st.latex(f"f(x)={pendiente}X {operador}{intercepto}")
        st.subheader("Predicción")
        indicadorPrediccion = "+ Positiva" if predict>=0 else "- Negativa"
        if(predict<=0):
            valuePredict = 0
        else:
            valuePredict = predict
        st.metric(f"El valor de la predicción de Muertes para {daysToPredict} dias desde el inicio de la pandemia es de: ",valuePredict, indicadorPrediccion)



        


else:
    st.warning("Debe Cargar un Archivo Previamente")
   


st.sidebar.title("Indice")
st.sidebar.markdown("### [Carga del Archivo](#carga-del-archivo)")
st.sidebar.markdown("- [Contenido del Archivo](#contenido-del-archivo)")
st.sidebar.markdown("### [Parametrización](#parametrizaci-n)")
st.sidebar.markdown("- [Variable Indepentiente (X)](#variable-independiente-x)")
st.sidebar.markdown("- [Variable Depentiente (Y)](#variable-dependiente-y)")
st.sidebar.markdown("- [Grado de la Función](#grado-de-la-funci-n)")
st.sidebar.markdown("- [Valor de la Predicción](#valor-de-la-predicci-n)")
st.sidebar.markdown("- [Colores de la Gráfica](#colores-de-la-gr-fica)")
st.sidebar.markdown("### [Graficación](#graficaci-n)")
st.sidebar.markdown("- [Datos de la Gráfica](#datos-de-la-gr-fica)")
st.sidebar.markdown("### [Función de la Tendencia](#funci-n-de-la-tendencia)")
st.sidebar.markdown("### [Predicción](#predicci-n)")


