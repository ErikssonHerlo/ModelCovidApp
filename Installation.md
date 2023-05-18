# Pasos de Instalación del Proyecto
## Requerimientos Previos
- Python _Version 3.8.10_
- Streamlit _Version 1.10.0_

## Instalación del Proyecto
Debemos levantar un entorno virtual en la carpeta con el comando
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Luego de tener nuestro entorno activado, debemos instalar las librerias que utilizaremos en nuestro proyecto
```bash
pip3 install sklearn
pip3 install scikit-learn 
pip3 install numpy 
pip3 install pandas 
pip3 install matplotlib
pip3 install streamlit
pip3 install --upgrade pip
```

## Encendido del Servidor
Luego de haber instalado de manera correcta las librerias que necesitamos para nuestro proyecto, levantaremos el servidor con el comando:
```bash
streamlit run Home_Page.py
```

## Apagado del Servidor
Para apacar el servidor, solo debemos de dar **Ctrl + c** en la terminal
