# Importación de bibliotecas necesarias
from sklearn.neighbors import KNeighborsRegressor  # Algoritmo KNN para regresión
from sklearn.metrics import mean_squared_error     # Para calcular el error del modelo
import matplotlib.pyplot as plt                    # Para visualizar los datos
import numpy as np                                 # Biblioteca para cálculos numéricos
import pandas as pd                                # Para manejar datos tabulares
import os                                          # Para interactuar con el sistema operativo

# Carga y preparación de datos
DATA_PATH = "C:\\Users\\kp\\OneDrive\\Escritorio\\Digital Nao\\Ciclo 1\\Notebooks"
FILE_BIKERPRO = 'SeoulBikeData.csv'
bikerpro = pd.read_csv(os.path.join(DATA_PATH, FILE_BIKERPRO), encoding="ISO-8859-1")  # Lee el archivo CSV

# Proceso para normalizar los nombres de las columnas
raw_columns = list(bikerpro.columns)  # Almacena los nombres originales de las columnas

# Genera nombres de columnas limpios y normalizados
clean_columns = [
    x.lower().replace("(°c)", '').replace("(%)", '').replace(" (m/s)", '')
    .replace(" (10m)", '').replace(" (mj/m2)", '').replace("(mm)", '')
    .replace(" (cm)", '').replace(" ", '_') for x in bikerpro.columns
]

# Actualiza los nombres de las columnas del DataFrame
bikerpro.columns = clean_columns

# Convierte la columna de fecha a formato datetime para un manejo más fácil
bikerpro['date'] = pd.to_datetime(bikerpro['date'], format='%d/%m/%Y')

# Selección de características (clima) y la variable objetivo (conteo de bicicletas)
weather_cols = ['temperature', 'humidity', 'wind_speed', 'visibility',
                'dew_point_temperature', 'solar_radiation', 'rainfall', 'snowfall']
target_col = ['rented_bike_count']

# Preparación del DataFrame para el modelado
X = bikerpro[weather_cols + target_col]
X = bikerpro.sort_values(['date', 'hour'])  # Ordena los datos por fecha y hora

# División de los datos en conjuntos de entrenamiento y prueba
X_train = X.loc[: X.shape[0]-1440, weather_cols]
y_train = X.loc[: X.shape[0]-1440, target_col]
X_test = X.loc[X.shape[0]-1440+1:, weather_cols]
y_test = X.loc[X.shape[0]-1440+1:, target_col]

# Modelado con KNN y evaluación de diferentes valores de 'k'
k_values = [3, 5, 10, 15, 20, 50, 100, 300, 500, 1000]
rmse_values = {}
for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    rmse_values[k] = np.sqrt(mean_squared_error(y_test, y_test_pred))  # Calcula el RMSE

# Identifica el valor de 'k' que minimiza el RMSE
best_k = min(rmse_values, key=rmse_values.get)

# Visualización de RMSE vs. valores de 'k'
plt.figure(figsize=(10, 6))
plt.plot(list(rmse_values.keys()), list(rmse_values.values()), marker='o')
plt.title('Error RMSE vs Valores de K')
plt.xlabel('Valor de K')
plt.ylabel('Error RMSE')
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Mejor K: {best_k}')
plt.legend()
plt.savefig('error_knn_weather.png')  # Guarda la gráfica
plt.show()  # Muestra la gráfica
