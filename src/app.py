# # LINDA VASQUEZ Sistemas de Recomendación

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

url = "https://breathecode.herokuapp.com/asset/internal-link?id=2326&path=adult-census-income.csv"
datos = pd.read_csv(url)

datos.replace("?", pd.NA, inplace=True)
datos.dropna(inplace=True)

datos['income'] = datos['income'].str.strip()

# variable objetivo (1 si gana más de 50K, 0 si gana menos o igual)
datos['alto_ingreso'] = datos['income'].apply(lambda x: 1 if x == '>50K' else 0)

# columnas importantes
columnas = ['age', 'education', 'marital.status', 'occupation', 'hours.per.week', 'sex', 'native.country']
X = datos[columnas]
y = datos['alto_ingreso']

# cuáles son números y cuáles son texto
numericas = ['age', 'hours.per.week']
categoricas = ['education', 'marital.status', 'occupation', 'sex', 'native.country']

# StandardScaler pone los números en la misma escala
trans_num = StandardScaler()
# OneHotEncoder convierte el texto en columnas de ceros y unos
trans_cat = OneHotEncoder(handle_unknown='ignore')

preprocesador = ColumnTransformer(
    transformers=[
        ("numeros", trans_num, numericas),
        ("categorias", trans_cat, categoricas),
    ])

modelo = Pipeline(steps=[
    ('limpieza', preprocesador),
    ('clasificador', LogisticRegression(max_iter=1000))
])

# 80% para entrenar y 20% para probar
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)

modelo.fit(X_entrenamiento, y_entrenamiento)

# funcion para nuestro Sistema de recomendacion
def recomendar_futuro(perfil):
    # diccionario a un DataFrame de una sola fila
    df_usuario = pd.DataFrame([perfil])
    
    # si gana más de 50K (da 0 o 1)
    prediccion = modelo.predict(df_usuario)[0]
    
    probabilidad = modelo.predict_proba(df_usuario)[0][1]
    porcentaje = probabilidad * 100
    
    #recomendación basada en el resultado
    if prediccion == 1:
        return f"Tu probabilidad de ganar más de 50K al año es del {porcentaje:.1f}%"
    else:
        return f"Tu probabilidad de superar los 50K es solo del {porcentaje:.1f}%. Intenta mejorar tu nivel educativo o buscar oportunidades en otra ocupacion."

#caso simulado
usuario_prueba = {
    "age": 25,
    "education": "HS-grad",
    "marital.status": "Never-married",
    "occupation": "Sales",
    "hours.per.week": 30,
    "sex": "Male",
    "native.country": "United-States"
}

resultado = recomendar_futuro(usuario_prueba)
print(resultado)