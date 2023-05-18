#Importar las librerias necesarias

import streamlit as st

st.title("Data Science")
st.write("""La ciencia de datos es el campo de la aplicación de técnicas analíticas avanzadas y principios científicos para extraer información valiosa de los datos para la toma de decisiones comerciales, la planificación estratégica y otros usos. Es cada vez más crítico para las empresas: la información que genera la ciencia de datos ayuda a las organizaciones a aumentar la eficiencia operativa, identificar nuevas oportunidades comerciales y mejorar los programas de marketing y ventas, entre otros beneficios. En última instancia, pueden generar ventajas competitivas sobre los rivales comerciales.
""")
st.subheader("Un recurso por descubrir para el aprendizaje automático")
st.write("""La Data science es uno de los campos más apasionantes que existen en la actualidad. ¿Por qué es tan importante?

Porque las empresas están sentadas sobre un tesoro de datos. Ya que la tecnología moderna ha permitido la creación y almacenamiento de cantidades cada vez mayores de información, los volúmenes de datos se han incrementado. Se estima que el 90% de los datos en el mundo se crearon en los últimos dos años. Por ejemplo, los usuarios de Facebook suben 10 millones de fotos por hora.

Pero a menudo estos datos se almacenan en bases de datos y lagos de datos, en su mayoría intactos.

La gran cantidad de datos recopilados y almacenados por estas tecnologías puede generar beneficios transformadores para las organizaciones y sociedades de todo el mundo, pero solo si sabemos interpretarlos. Ahí es donde entra en acción la ciencia de datos.

La ciencia de datos revela tendencias y genera información que las empresas pueden utilizar para tomar mejores decisiones y crear productos y servicios más innovadores. Quizás lo más importante es que permite que los modelos de aprendizaje automático extraigan conocimientos de las grandes cantidades de datos que se les suministran, evitando así depender principalmente de los analistas empresariales para ver qué pueden descubrir a partir de los datos.

Los datos son los cimientos de la innovación, pero su valor proviene de la información que los científicos de datos pueden extraer de ellos y luego utilizar.
""")
st.subheader("Aplicación")
st.write("Por ende, esta aplicación esta construida con la finalidad de visualizar las distintas aplicaciones que tiene la Ciencia de Datos. Por ello cuenta con los siguientes algoritmos y operaciones:")
st.markdown("#### Algoritmos")
st.markdown("""
- Regresión Lineal
- Regresión Polinomial

""")
st.markdown("#### Operaciones")
st.markdown("""
- Graficación de puntos
- Graficación de la tendencia
- Definición de función de tendencia lineal
- Definición de función de tendencia polinomial
- Predicción de la tendencia (según la unidad de tiempo ingresada)
""")





st.sidebar.title("Bienvenidos")

st.sidebar.markdown("""
## Ingenieria CUNOC - USAC
### Laboratorio de Modelación y Simulación 1
Ing. Pedro Domingo
""")

st.sidebar.markdown("""
## Integrantes:
- #1 Eriksson José Hernández López - 201830459
- #10 César Reginaldo Tzoc Alvarado - 201430927
- #16 Juan Pablo Meza Vielman - 201930268
""")

st.sidebar.markdown("@ErikssonHerlo on "+
    '<a href="https://github.com/ErikssonHerlo/DataScience" target="_blank">GitHub</a>', unsafe_allow_html=True)


