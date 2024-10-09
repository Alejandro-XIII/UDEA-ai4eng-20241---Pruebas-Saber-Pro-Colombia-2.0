import argparse
import logging
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configurar el logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Función para cargar los datos
def load_data(file_path):
    logger.info("Loading train data")
    data = pd.read_csv(file_path)
    logger.info("Data types:\n%s", data.dtypes)
    return data

# Función para convertir las columnas categóricas a numéricas
def preprocess_data(df):
    logger.info("Converting categorical columns")
    
    # Convertir todas las columnas categóricas a numéricas
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    
    return df, label_encoders

# Función para entrenar el modelo
def train_model(X_train, y_train, model_file, overwrite_model=False):
    # Si el modelo ya existe y no queremos sobrescribir, cargar el modelo
    if os.path.exists(model_file) and not overwrite_model:
        logger.info("Loading existing model from %s", model_file)
        model = xgb.XGBClassifier()
        model.load_model(model_file)
    else:
        logger.info("Fitting model")
        model = xgb.XGBClassifier()
        model.fit(X_train, y_train)
        
        logger.info("Saving model to %s", model_file)
        # Guardar el modelo en formato JSON
        model.get_booster().save_model(model_file)

    return model

# Función principal
def main(data_file, model_file, overwrite_model):
    # Cargar los datos
    data = load_data(data_file)

    # Convertir las columnas categóricas y la columna objetivo
    data, label_encoders = preprocess_data(data)

    # Separar características (X) y variable objetivo (y)
    X = data.drop(columns='RENDIMIENTO_GLOBAL')
    y = data['RENDIMIENTO_GLOBAL']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = train_model(X_train, y_train, model_file, overwrite_model)

if __name__ == "__main__":
    # Argumentos para el script
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the training data file")
    parser.add_argument('--model_file', type=str, required=True, help="Path to save the trained model")
    parser.add_argument('--overwrite_model', action='store_true', help="Overwrite existing model")
    
    args = parser.parse_args()

    # Crear el directorio para guardar el modelo si no existe
    if not os.path.exists(os.path.dirname(args.model_file)):
        os.makedirs(os.path.dirname(args.model_file))

    main(args.data_file, args.model_file, args.overwrite_model)