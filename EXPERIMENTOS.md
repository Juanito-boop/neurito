# Experimentos de Neurito

Los experimentos usan 3000 ejemplos sintéticos para entrenar y 1000 para
probar. Cada entrada tiene tres valores entre 0 y 1, en este orden:
`[hambre, energía, aburrimiento]`. La etiqueta se calcula con la política:

```text
comer      = hambre
dormir     = 1 - energía
jugar      = aburrimiento
```

La etiqueta final es la acción con el valor mayor. Se mantuvieron la semilla
42, el descenso por gradiente de lote completo y la tasa 0.1, salvo cuando se
indica lo contrario. La red ahora permite cambiar estos parámetros mediante
`train_network` en [core.py](/home/v/pruebas_ia/ia/neurito/src/neurito/core.py).

## 1. Dos necesidades fijas y variación de la tercera

Se fijaron hambre en 0.6 y aburrimiento en 0.2, y se varió únicamente la
energía:

| Energía | Probabilidades `[comer, dormir, jugar]` | Acción |
|---:|---|---|
| 0.0 | `[0.0029, 0.9971, 0.0000]` | dormir |
| 0.3 | `[0.1115, 0.8884, 0.0001]` | dormir |
| 0.4 | `[0.4431, 0.5565, 0.0004]` | dormir |
| 0.5 | `[0.8341, 0.1652, 0.0008]` | comer |
| 0.8 | `[0.9984, 0.0008, 0.0008]` | comer |
| 1.0 | `[0.9991, 0.0000, 0.0009]` | comer |

La acción predicha cambia entre 0.4 y 0.5 de energía: al aumentar la energía,
`1 - energía` deja de superar a hambre. El tercer valor queda bajo y no llega
a competir.

## 2. Prioridades casi iguales

Con entradas cercanas a `[0.5, 0.5, 0.5]`, pequeños cambios producen cambios
visibles en Softmax:

| Entrada `[hambre, energía, aburrimiento]` | `[comer, dormir, jugar]` | Acción |
|---|---|---|
| `[0.50, 0.50, 0.50]` | `[0.3034, 0.3304, 0.3662]` | jugar |
| `[0.51, 0.50, 0.50]` | `[0.3437, 0.3115, 0.3448]` | jugar |
| `[0.50, 0.49, 0.50]` | `[0.2839, 0.3719, 0.3442]` | dormir |
| `[0.50, 0.50, 0.51]` | `[0.2836, 0.3075, 0.4089]` | jugar |

Softmax no devuelve una decisión binaria: reparte probabilidad entre las
acciones. Cuando las prioridades son cercanas, una variación de 0.01 puede
cambiar el máximo, aunque las tres probabilidades sigan siendo comparables.

## 3. Tamaño de la capa oculta

Evaluación con 1200 épocas y tasa 0.1:

| Neuronas ocultas | Pérdida final de entrenamiento | Precisión de prueba |
|---:|---:|---:|
| 3 | 0.1629 | 97.2% |
| 6 | 0.1365 | 98.1% |
| 10 | 0.1541 | 98.2% |
| 20 | 0.1280 | 99.1% |

Tres neuronas tienen menos capacidad. Más neuronas no garantizan una pérdida
menor en todas las semillas, como muestra el caso de 10, pero 20 representa
mejor esta frontera de decisión.

## 4. Épocas y tasa de aprendizaje

| Épocas | Tasa | Pérdida final | Precisión |
|---:|---:|---:|---:|
| 100 | 0.1 | 0.8914 | 82.1% |
| 1200 | 0.1 | 0.1365 | 98.1% |
| 3000 | 0.1 | 0.0813 | 98.9% |
| 1200 | 0.01 | 0.8285 | 82.9% |
| 1200 | 0.5 | 0.0675 | 96.7% |
| 1200 | 1.0 | 0.0745 | 96.2% |

Con 100 épocas el modelo está subentrenado; con 3000 todavía mejora. La tasa
0.01 avanza demasiado despacio. Tasas 0.5 y 1.0 reducen la pérdida de
entrenamiento rápido, pero generalizan peor que 0.1 en esta prueba, por lo que
el aprendizaje empieza a ser menos estable. En estas corridas no apareció
`NaN` ni divergencia; una tasa aún mayor podría volverlo inestable.

## 5. ¿Qué datos y etiquetas requiere una cuarta acción?

Primero hay que definir la acción y su regla. Por ejemplo, para añadir
`explorar`, se necesitarían ejemplos donde la etiqueta 3 sea `explorar` cuando
la combinación de necesidades lo justifique (por ejemplo, aburrimiento medio
con hambre baja y energía alta). Cada ejemplo debe contener las mismas tres
entradas normalizadas y una etiqueta entera:

```text
[hambre, energía, aburrimiento] -> 0 comer, 1 dormir, 2 jugar, 3 explorar
```

La salida de la red tendría que cambiar de tres a cuatro neuronas (`W2` y
`bias2`), `actions` tendría que incluir `"explore"` y habría que agregar
imágenes y texto de interfaz. Conviene generar suficientes ejemplos de la
nueva región y reservar ejemplos de prueba para las cuatro clases; de lo
contrario, la cuarta acción podría quedar infrarrepresentada.

## Explicación de los resultados

### 1. ¿En qué se diferencian pesos y sesgos?

Los **pesos** multiplican las entradas y determinan cuánto influye cada
conexión. Los **sesgos** son valores ajustables que se suman antes de la
activación; permiten desplazar el umbral de una neurona incluso cuando sus
entradas son cero. En el código, `weights1`/`weights2` son pesos y
`bias1`/`bias2` son sesgos.

### 2. ¿Por qué el cansancio es `1 - energía`?

La interfaz guarda energía alta como un valor grande (`1` significa mucha
energía), pero dormir debe ser prioritario cuando la energía es baja. La
transformación `1 - energía` invierte la escala: energía 0 produce cansancio
1 y energía 1 produce cansancio 0. Así la política conserva una escala común
de “necesidad alta = prioridad alta”.

### 3. ¿Qué hace ReLU y qué hace Softmax?

**ReLU** devuelve `max(0, x)`: elimina valores negativos y conserva los
positivos. Introduce una no linealidad que permite que la capa oculta aprenda
fronteras más útiles que una transformación lineal única.

**Softmax** convierte los tres logits finales en valores positivos que suman
1. Se interpretan como probabilidades relativas; `argmax` selecciona la
acción con mayor probabilidad.

### 4. ¿Por qué una precisión alta puede ser engañosa?

La precisión solo cuenta cuántas etiquetas fueron acertadas y puede ocultar
una clase minoritaria ignorada, datos de prueba demasiado parecidos a los de
entrenamiento o una política sintética demasiado sencilla. También puede
ocultar predicciones poco confiables cerca de un empate. Por eso se deben
revisar pérdida, probabilidades por clase, distribución de etiquetas y
ejemplos de prueba representativos, no solo el porcentaje global.
