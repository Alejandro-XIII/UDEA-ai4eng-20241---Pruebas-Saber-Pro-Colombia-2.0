import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import sys

def predict():
    def log(message):
        print(message)
        sys.stdout.flush()  # Asegura que el mensaje se envíe inmediatamente al cliente

    log("Cargando el modelo entrenado...")
    model = joblib.load('./data/xgb_model.pkl')

    log("Cargando los nuevos datos desde el archivo CSV...")
    df_new = pd.read_csv('./data/test.csv')

    log("Rellenando valores faltantes con la moda de cada columna...")
    columns_with_missing_values = [
        'ESTU_VALORMATRICULAUNIVERSIDAD', 'ESTU_HORASSEMANATRABAJA',
        'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEINTERNET',
        'FAMI_EDUCACIONPADRE', 'FAMI_TIENELAVADORA',
        'FAMI_TIENEAUTOMOVIL', 'ESTU_PAGOMATRICULAPROPIO',
        'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET.1',
        'FAMI_EDUCACIONMADRE'
    ]
    mode_values = {column: df_new[column].mode()[0] for column in columns_with_missing_values}
    df_new.fillna(value=mode_values, inplace=True)

    log("Aplicando la codificación one-hot a las columnas categóricas...")
    columns_to_exclude = ['ID', 'PERIODO', 'ESTU_PRGM_ACADEMICO']
    columns_to_encode = [col for col in df_new.columns if col not in columns_to_exclude]
    df_new_pro = pd.get_dummies(df_new[columns_to_encode]).astype(int)

    log("Concatenando y normalizando las columnas numéricas...")
    df_new_pro = pd.concat([df_new[['ID', 'PERIODO', 'ESTU_PRGM_ACADEMICO']], df_new_pro], axis=1)
    encoder = LabelEncoder()
    df_new_pro['ESTU_PRGM_ACADEMICO'] = encoder.fit_transform(df_new_pro['ESTU_PRGM_ACADEMICO'])
    scaler = MinMaxScaler()
    df_new_pro['PERIODO'] = scaler.fit_transform(df_new_pro[['PERIODO']])
    df_new_pro['ESTU_PRGM_ACADEMICO'] = scaler.fit_transform(df_new_pro[['ESTU_PRGM_ACADEMICO']])

    log("Realizando predicciones con el modelo cargado...")
    y_pred = model.predict(df_new_pro)
    y_pred = pd.Series(y_pred).map({
        0: "bajo", 
        1: "medio-bajo", 
        2: "medio-alto", 
        3: "alto"
    })

    log("Guardando los resultados en un archivo CSV...")
    submission = pd.DataFrame({
        'ID': df_new_pro['ID'], 
        'RENDIMIENTO_GLOBAL': y_pred
    })
    submission.to_csv('./data/submission.csv', index=False)

    log("Predicciones guardadas exitosamente en data/submission.csv.")

# Llamar a la función si se ejecuta directamente este script
if __name__ == "__main__":
    predict()
