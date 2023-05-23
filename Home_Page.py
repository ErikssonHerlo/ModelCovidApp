#Importar las librerias necesarias

import streamlit as st

st.title("Modelación y Simulación con Modelos Matemáticos")
st.write("""La Modelación y Simulación utilizando Modelos Matemáticos es una poderosa herramienta que nos permite comprender y predecir el comportamiento de fenómenos complejos en diversos campos. Mediante la formulación de ecuaciones y relaciones matemáticas que describen el sistema en estudio, podemos obtener una representación simplificada pero precisa de su funcionamiento. Estos modelos nos permiten realizar simulaciones y experimentos virtuales, lo que nos brinda la capacidad de explorar diferentes escenarios y evaluar el impacto de cambios en las variables de interés. La Modelación y Simulación basada en Modelos Matemáticos es una disciplina fundamental para el avance científico y tecnológico, ya que nos proporciona una herramienta invaluable para tomar decisiones informadas y optimizar procesos en diversas áreas de estudio.""")
st.subheader("El uso de la Ciencia de Datos en la Modelación de Modelos Matemáticos")
st.write("""La Ciencia de Datos ha revolucionado numerosos campos, y su aplicación en la Modelación y Simulación no es una excepción. La Modelación y Simulación son herramientas fundamentales para comprender y predecir fenómenos complejos en diversas disciplinas, como la física, la biología, la economía y la ingeniería. Sin embargo, la incorporación de la Ciencia de Datos en este ámbito ha permitido un enfoque más preciso y sofisticado.

La Ciencia de Datos se centra en la extracción de conocimientos a partir de grandes volúmenes de datos, utilizando técnicas de análisis estadístico y aprendizaje automático. Al aplicar estos métodos en la Modelación y Simulación, es posible obtener una visión más completa y detallada de los sistemas que se están estudiando.

La utilización de datos reales en la modelación y simulación permite una mayor fidelidad y precisión en la representación de los fenómenos. La recopilación de datos relevantes y su preprocesamiento adecuado son pasos fundamentales para garantizar resultados confiables. Además, el análisis exploratorio de datos revela patrones y tendencias ocultas, proporcionando información valiosa para la construcción de modelos más realistas.

La construcción de modelos basados en técnicas de aprendizaje automático permite capturar las complejidades de los sistemas en estudio. Estos modelos pueden aprender de los datos, identificar relaciones entre variables y realizar predicciones más precisas. La simulación y experimentación con estos modelos ofrecen la posibilidad de explorar diferentes escenarios y condiciones, proporcionando una comprensión más profunda de los sistemas y permitiendo la toma de decisiones fundamentadas.

En resumen, la Ciencia de Datos ha brindado un enfoque innovador en la Modelación y Simulación, permitiendo un análisis más riguroso y preciso de fenómenos complejos. La integración de datos, análisis estadístico y aprendizaje automático ha llevado a un avance significativo en nuestra capacidad para comprender, predecir y tomar decisiones informadas en diversos campos de estudio.""")

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

st.title("Regresión Polinomial")
st.write("""
La Regresión Polinomial es un caso especial de la Regresión Lineal, extiende el modelo lineal al agregar predictores adicionales, obtenidos al elevar cada uno de los predictores originales a una potencia. Por ejemplo, una regresión cúbica utiliza tres variables, como predictores. Este enfoque proporciona una forma sencilla de proporcionar un ajuste no lineal a los datos.
""")
st.write("""
El método estándar para extender la Regresión Lineal a una relación no lineal entre las variables dependientes e independientes, ha sido reemplazar el modelo lineal con una función polinomial.
""")
st.write("""
Por su parte, la ecuación general correspondiente a un modelo de regresión polinomial es:
""")
st.latex("Y=β0+β1Xi+βnXi^n+ϵi")

st.write("""
Como se puede observar para la Regresión Polinomial se crean algunas características adicionales que no se encuentran en la Regresión Lineal.

Un término polinomial, bien sea cuadrático o cúbico, convierte un modelo de regresión lineal en una curva, pero como los datos de “X” son cuadráticos o cúbicos pero el coeficiente “b” no lo es, todavía se califican como un modelo lineal.

Esto hace que sea una forma agradable y directa de modelar curvas sin tener que modelar modelos complicados no lineales.
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
- Definición de función de tendencia polinomial
- Predicción de la tendencia (según la unidad de tiempo ingresada)
- Predicción de la tendencia de Vacunación(según el tipo de vacuna, numero de dosis y unidad de tiempo ingresada)
- Indice de Progresión Epidémiologica (EPI)
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


