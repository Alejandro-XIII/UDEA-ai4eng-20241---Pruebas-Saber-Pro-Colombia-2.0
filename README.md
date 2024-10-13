# UDEA-ai4eng-20241---Pruebas-Saber-Pro-Colombia-2.0

Segunda parte del modelo de clasificación para predecir el desempeño de estudiantes en las Pruebas Saber Pro de Colombia. En esta ocasión, el modelo se lleva a un estado listo para ser integrado en un sistema de producción utilizando Docker.

## FASE 1. Modelo Predictivo:

1. **Descargar los Archivos**  
   Descarga los archivos necesarios desde el repositorio. Asegúrate de tener los archivos CSV de la competencia de [Kaggle](https://www.kaggle.com/competitions/udea-ai4eng-20241).

2. **Cargar todos los archivos en Google Colab y Drive**  
   Sube los archivos CSV a Google Colab y Google Drive para su procesamiento.

3. **Reemplazar la ruta del archivo**  
   En el código proporcionado en el cuaderno `modelo_solucion.ipynb`, reemplaza las rutas de los archivos CSV con las correspondientes a Google Drive.

4. **Ejecutar el código en el orden indicado**  
   Sigue el flujo del cuaderno para entrenar y evaluar el modelo.

## FASE 2. Despliegue en container:

1. **Descargar los archivos `train.csv` y `test.csv`**  
   Descarga los archivos de entrenamiento y prueba desde la misma página de la fase 1 en [Kaggle](https://www.kaggle.com/competitions/udea-ai4eng-20241) y colócalos en la carpeta `fase-2/data`.

2. **Construir la imagen Docker**  
   Abre una terminal en la carpeta `fase-2` y ejecuta el siguiente comando para construir la imagen Docker:
   ```bash
   docker build -t my_model .
3. **Ejecutar el contenedor Docker**  
   Corre el siguiente comando para crear e iniciar el contenedor Docker:
   ```bash
   docker run -it my_model
4. **Entrenar o predecir**  
   Una vez dentro de la consola del contenedor, puedes ejecutar los scripts de entrenamiento o predicción:
   - Para entrenar el modelo:

      ```bash
      python scripts/train.py
   - Para realizar una predicción:

      ```bash
      python scripts/predict.py
5. **Verificar archivos generados'**  
   Para verificar que el modelo entrenado y el archivo de salida submission.csv fueron creados correctamente, ejecuta:

    ```bash
   ls /app/data
6. **Finalizar el contenedor**
   Cuando hayas terminado, puedes salir del contenedor con:
    ```bash
   exit
## Autor
-Diego Alejandro Castañeda Ossa - [Alejandro-XIII](https://github.com/Alejandro-XIII)
-CC 1036656438
-Ingeniería de Sistemas
