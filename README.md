# Pneumonia Detection

Clasificación de radiografías de tórax con **deep learning** mediante *transfer learning* (VGG16 preentrenada en ImageNet), con Keras/TensorFlow. Proyecto académico con dos modelos:

| Modelo | Tarea | Clases | Accuracy en test |
|---|---|---|---|
| 1. Normal vs. neumonía | Detectar si hay neumonía | NORMAL, PNEUMONIA | **92,47 %** |
| 2. Normal / bacteriana / viral | Distinguir además el origen de la neumonía | NORMAL, BACTERIAL, VIRAL | **84,94 %** |


## Contenido

| Fichero | Descripción |
|---|---|
| `Chest_Xray.ipynb` | Modelo binario: normal vs. neumonía |
| `Chest_Xray_Viral-Bacterial.ipynb` | Modelo de 3 clases: normal, neumonía bacteriana y neumonía viral |

Los notebooks están pensados para ejecutarse en **Google Colab** (usan Google Drive para el dataset y para guardar los pesos).

## Dataset

[Chest X-Ray Images (Pneumonia)](https://data.mendeley.com/datasets/rscbjbr9sj/2), de Kermany et al. (2018), publicado en [Cell](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5). Radiografías de tórax pediátricas ya divididas en `train`, `val` y `test`.

| Conjunto | NORMAL | PNEUMONIA | de ellas BACTERIAL | de ellas VIRAL |
|---|---|---|---|---|
| Train | 1342 | 3876 | 2531 | 1346 |
| Test | 234 | 390 | 242 | 149 |

Observaciones sobre los datos:
- **Clases desbalanceadas:** hay casi el triple de imágenes de neumonía que normales.
- **Validación muy pequeña:** la carpeta `val` original tiene solo 18 imágenes, así que se reserva un **20 % de `train`** como validación (4173 imágenes de entrenamiento y 1043 de validación en el modelo binario; 4180 y 1044 en el de 3 clases).
- Para el modelo de 3 clases se reorganizan las carpetas para que BACTERIAL y VIRAL queden al mismo nivel que NORMAL.

## Metodología

### Preparación de los datos
- Imágenes redimensionadas a **224×224** (tamaño de entrada de VGG16) y normalizadas (`rescale=1/255`).
- **Data augmentation** solo en entrenamiento, para que el modelo no memorice las imágenes: rotación (10°), desplazamientos (5 %), zoom (10 %) y volteo horizontal.
- Batch de 16 imágenes y `seed=42` para que la división 80/20 sea reproducible.
- **Pesos de clase** (`class_weight='balanced'`) para compensar el desbalance.

### Arquitectura
Base convolucional **VGG16** (pesos de ImageNet, sin la parte final) + clasificador propio. Solo se reentrenan las últimas capas convolucionales, porque las primeras detectan patrones simples (bordes, líneas) que ya sirven.

| | Modelo 1 (binario) | Modelo 2 (3 clases) |
|---|---|---|
| Capas de VGG16 reentrenadas | `block5` | `block4` y `block5` |
| Clasificador | Flatten → Dense(256) → Dropout(0,5) → Dense(1, sigmoid) | Flatten → Dense(512) → Dropout(0,5) → Dense(256) → Dropout(0,3) → Dense(3, softmax) |
| Pérdida | `binary_crossentropy` | `categorical_crossentropy` |
| Optimizador | RMSprop, lr = 1e-4 | RMSprop, lr = 5e-5 |
| Parámetros totales / entrenables | 21,1 M / 13,5 M | 27,7 M / 26,0 M |
| Pesos de clase | NORMAL 1,94 · PNEUMONIA 0,67 | BACTERIAL 0,69 · NORMAL 1,29 · VIRAL 1,29 |

En el modelo 2 se descongelan más capas y se baja el learning rate porque distinguir neumonía viral de bacteriana es visualmente más difícil y requiere más capacidad, pero con un ajuste más fino para no destruir los pesos preentrenados.

### Entrenamiento
Hasta 30 epochs con dos callbacks:
- **`ModelCheckpoint`:** guarda el mejor modelo según `val_acc`.
- **`EarlyStopping`:** detiene el entrenamiento si `val_acc` no mejora en 10 epochs.

La evaluación final en test se hace cargando los mejores pesos guardados, no los del último epoch.

## Resultados

- **Modelo 1 (normal vs. neumonía):** 92,47 % de accuracy en el conjunto de test (624 imágenes). Las curvas de entrenamiento y validación se mantienen cercanas, sin señales claras de overfitting.
- **Modelo 2 (3 clases):** 84,94 % de accuracy en test (624 imágenes), con un mejor `val_acc` de 77,3 % durante el entrenamiento. Las curvas de validación oscilan bastante y hay síntomas de overfitting, por lo que el *early stopping* fue necesario para quedarse con los mejores pesos. El problema es claramente más difícil que el binario, ya que neumonía viral y bacteriana se parecen mucho en radiografía.

## Limitaciones y posibles mejoras

- **Solo se reporta accuracy.** En un problema médico con clases desbalanceadas interesan también la *sensibilidad (recall)* de la neumonía, la precisión y la matriz de confusión. Sería el siguiente paso.
- **Posible fuga de datos en la validación:** la validación se separa al azar del conjunto de train, y si hay varias radiografías de un mismo paciente pueden aparecer a ambos lados. Por eso las métricas de validación podrían ser optimistas. El conjunto de test sí es independiente.
- **Dataset pediátrico de un único origen:** el modelo no está validado con otros hospitales, equipos ni poblaciones.
- Mejoras posibles: probar otras arquitecturas (ResNet, EfficientNet), técnicas de interpretabilidad como Grad-CAM para ver en qué zonas se fija el modelo, y *fine-tuning* más cuidadoso del modelo de 3 clases.

## Cómo ejecutarlo

1. Descarga el dataset (enlace de arriba) y súbelo a tu Google Drive. Para el modelo 2 hace falta la versión en la que la carpeta `PNEUMONIA` está dividida en `BACTERIAL` y `VIRAL`.
2. Abre el notebook en Google Colab y monta tu Drive.
3. Ajusta las rutas del dataset y de guardado de pesos (`/content/drive/MyDrive/...`) a las tuyas.
4. Ejecuta las celdas en orden.

Requisitos: Python 3, TensorFlow/Keras, scikit-learn, NumPy y Matplotlib (ya incluidos en Colab).

## Tecnologías

Python · TensorFlow / Keras · VGG16 · transfer learning · data augmentation · scikit-learn · Matplotlib · Google Colab

## Autores

Francisco Ferrer López
