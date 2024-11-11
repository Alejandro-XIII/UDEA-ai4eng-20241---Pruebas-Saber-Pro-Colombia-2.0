import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib

def train_model():
    print("Importando librerías y cargando los datos...")
    df_train = pd.read_csv('./data/train.csv')

    print("Rellenando valores faltantes con la moda de cada columna...")
    columns_with_missing_values = [
        'ESTU_VALORMATRICULAUNIVERSIDAD', 'ESTU_HORASSEMANATRABAJA',
        'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEINTERNET',
        'FAMI_EDUCACIONPADRE', 'FAMI_TIENELAVADORA',
        'FAMI_TIENEAUTOMOVIL', 'ESTU_PAGOMATRICULAPROPIO',
        'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET.1',
        'FAMI_EDUCACIONMADRE'
    ]
    mode_values = {column: df_train[column].mode()[0] for column in columns_with_missing_values}
    df_train.fillna(value=mode_values, inplace=True)

    print("Aplicando la codificación one-hot a las columnas categóricas...")
    columns_to_exclude = ['ID', 'PERIODO', 'ESTU_PRGM_ACADEMICO', 'RENDIMIENTO_GLOBAL']
    columns_to_encode = [col for col in df_train.columns if col not in columns_to_exclude]
    df_train_pro = pd.get_dummies(df_train[columns_to_encode]).astype(int)

    print("Agregando la columna objetivo 'RENDIMIENTO_GLOBAL' al dataset procesado...")
    df_train_pro['RENDIMIENTO_GLOBAL'] = df_train['RENDIMIENTO_GLOBAL']

    print("Cambiando el target a numérico...") 
    df_train_pro['RENDIMIENTO_GLOBAL'] = df_train_pro['RENDIMIENTO_GLOBAL'].map({
        "bajo": 0, "medio-bajo": 1, "medio-alto": 2, "alto": 3
    })

    print("Concatenando y normalizando las columnas numéricas...")
    df_train_pro = pd.concat([df_train[['ID','PERIODO','ESTU_PRGM_ACADEMICO']], df_train_pro], axis=1)
    encoder = LabelEncoder()
    df_train_pro['ESTU_PRGM_ACADEMICO'] = encoder.fit_transform(df_train_pro['ESTU_PRGM_ACADEMICO'])
    scaler = MinMaxScaler()
    df_train_pro['PERIODO'] = scaler.fit_transform(df_train_pro[['PERIODO']])
    df_train_pro['ESTU_PRGM_ACADEMICO'] = scaler.fit_transform(df_train_pro[['ESTU_PRGM_ACADEMICO']])

    print("Dividiendo los datos en conjuntos de entrenamiento y prueba...")
    X = df_train_pro.drop('RENDIMIENTO_GLOBAL', axis=1)
    y = df_train_pro['RENDIMIENTO_GLOBAL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Entrenando el modelo...")
    model = xgb.XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    print("Evaluando el modelo...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisión del modelo: {accuracy * 100:.2f}%')

    print("Guardando el modelo...")
    joblib.dump(model, './data/xgb_model.pkl')
    print("Modelo guardado en data/xgb_model.pkl")

# Llamar a la función si se ejecuta directamente este script
if __name__ == "__main__":
    train_model()
