# Autores:

* Sebastian Ramirez Escobar
* Sebastian Carvalho Salazar
* Johan Stiven Paez Bermudez

# Taller 3 - Factorización de matrices

## TABLA CONTENIDO
Descriptcion Problema [https://github.com/stijopa/Taller_3/edit/main/README.md#problema-2]


## PROBLEMA 1
Lea sobre el método de potencias para aproximar el valor propio mayor de
una matríz. Realice los dos ejercicios que se encuentran al final de esta guía:
Approximating Eigenvalues  -  Jupyter Guide to Linear Algebra. (https://bvanderlei.github.io/jupyter-guide-to-linear-algebra/Approximating_Eigenvalues.html)

### Problema 1: Sea la matriz A del ejemplo del método de potencia inversa.

\begin{pmatrix}
  a & b\\ 
  c & d
\end{pmatrix}

### Problema 1.a: Utilice el método de la potencia para aproximar el valor propio más grande λ1. Verifique que el valor exacto de λ1 es 12.

## Solucion 1.a

```python
import numpy as np

# Define la matriz A
A = np.array([[9, -1, -3],
              [0, 6, 0],
              [-6, 3, 6]])


# Inicializa un vector inicial (puede ser aleatorio)
x = np.array([1, 1, 1])

# Número máximo de iteraciones
max_iterations = 100

# Tolerancia para la convergencia
tolerance = 0.000001


# Iteración del método de la potencia
for i in range(max_iterations):
    # Multiplica A por el vector actual
    Ax = np.dot(A, x)
    
    # Encuentra el valor propio aproximado
    eigenvalue_estimate = np.dot(x, Ax) / np.dot(x, x)
    
    # Normaliza el vector
    x = Ax / np.linalg.norm(Ax)
    
    # Comprueba la convergencia
    if i > 0 and np.abs(eigenvalue_estimate - prev_eigenvalue_estimate) < tolerance:
        break
    
    prev_eigenvalue_estimate = eigenvalue_estimate

# Imprime el resultado
print("Valor propio más grande aproximado (λ1):", eigenvalue_estimate)

# Verifica que el valor exacto de λ1 es 12
exact_lambda1 = 12
if np.isclose(eigenvalue_estimate, exact_lambda1, atol=tolerance):
    print("El valor propio aproximado es cercano al valor exacto de λ1 (12).")
else:
    print("El valor propio aproximado no es cercano al valor exacto de λ1 (12).")
```
### Problema 1.b:  Aplique el método de potencia inversa con un desplazamiento de μ =10. Explique por qué los resultados difieren de los del ejemplo.
## Solucion 1.b
```python
import numpy as np

# Define la matriz A
A = np.array([[9, -1, -3],
              [0, 6, 0],
              [-6, 3, 6]])

# Define el desplazamiento μ
mu = 10

# Obtiene las dimensiones de la matriz A
n = A.shape[0]

# Inicializa un vector aleatorio como aproximación inicial
x = np.random.rand(n)

# Número máximo de iteraciones
max_iterations = 100

# Tolerancia para la convergencia
tolerance = 0.000001

# Iteración del método de la potencia inversa con desplazamiento
for i in range(max_iterations):
    # Resuelve el sistema de ecuaciones lineales (A - μI) * y = x
    y = np.linalg.solve(A - mu * np.eye(n), x)
    
    # Normaliza el vector y
    y = y / np.linalg.norm(y)
    
    # Comprueba la convergencia
    if i > 0 and np.linalg.norm(y - x) < tolerance:
        break
    
    x = y

# Calcula el valor propio más pequeño
eigenvalue_estimate = 1 / np.dot(y, x) + mu

# Imprime el resultado
print("Valor propio más pequeño aproximado con desplazamiento (μ=10):", eigenvalue_estimate)


```

### Problema 1.c: Aplique el método de potencia inversa con un desplazamiento de μ=7.5 y el vector inicial que se muestra a continuación. Explique por qué la secuencia de vectores se aproxima al vector propio correspondiente a λ1

## Solucion 1.c
```python
import numpy as np

# Define la matriz A
A = np.array([[9, -1, -3],
              [0, 6, 0],
              [-6, 3, 6]])

# Define el desplazamiento μ
mu = 7.5

# Define el vector inicial proporcionado
x = np.array([1, 0, 0])

# Número máximo de iteraciones
max_iterations = 100

# Tolerancia para la convergencia
tolerance = 0.000001

# Lista para almacenar la secuencia de vectores
vector_sequence = []

# Iteración del método de la potencia inversa con desplazamiento
for i in range(max_iterations):
    # Resuelve el sistema de ecuaciones lineales (A - μI) * y = x
    y = np.linalg.solve(A - mu * np.eye(A.shape[0]), x)
    
    # Normaliza el vector y
    y = y / np.linalg.norm(y)
    
    # Agrega el vector actual a la secuencia
    vector_sequence.append(y)
    
    # Comprueba la convergencia
    if i > 0 and np.linalg.norm(y - x) < tolerance:
        break
    
    x = y

# Imprime la secuencia de vectores
print("Secuencia de vectores aproximados:")
for i, v in enumerate(vector_sequence):
    print(f"Iteración {i + 1}:", v)

# Imprime el último vector normalizado, que se acerca al vector propio correspondiente a λ1
print("\nVector propio correspondiente a λ1 aproximado (normalizado):", vector_sequence[-1])

```
## Problema 2: Sea ​​la matriz B.


### Problema 2.a: Aplicar el método de potencia y el método de potencia inversa con desplazamientos para aproximar todos los valores propios de la matriz B. (Tenga en cuenta que uno de los valores propios de esta matriz es negativo).

## Solucion 2.a
```python
import numpy as np

# Define la matriz B
B = np.array([[-2, -18, 6],
              [-11, 3, 11],
              [-27, 15, 31]])

# Función para el método de la potencia
def power_iteration(matrix, num_iterations):
    n = matrix.shape[0]
    
    # Inicializa un vector aleatorio como aproximación inicial
    x = np.random.rand(n)
    
    eigenvalues = []
    eigenvectors = []
    
    for _ in range(num_iterations):
        # Multiplica la matriz por el vector actual
        y = np.dot(matrix, x)
        
        # Encuentra el valor propio aproximado
        eigenvalue = np.dot(x, y) / np.dot(x, x)
        eigenvalues.append(eigenvalue)
        
        # Normaliza el vector y
        x = y / np.linalg.norm(y)
        eigenvectors.append(x)
    
    return eigenvalues, eigenvectors

# Función para el método de la potencia inversa con desplazamiento
def inverse_power_iteration(matrix, shift, num_iterations):
    n = matrix.shape[0]
    
    # Inicializa un vector aleatorio como aproximación inicial
    x = np.random.rand(n)
    
    eigenvalues = []
    eigenvectors = []
    
    for _ in range(num_iterations):
        # Resuelve el sistema de ecuaciones lineales (B - shift*I) * y = x
        y = np.linalg.solve(matrix - shift * np.eye(n), x)
        
        # Encuentra el valor propio aproximado
        eigenvalue = np.dot(x, y) / np.dot(x, x) + shift
        eigenvalues.append(eigenvalue)
        
        # Normaliza el vector y
        x = y / np.linalg.norm(y)
        eigenvectors.append(x)
    
    return eigenvalues, eigenvectors

# Número máximo de iteraciones
num_iterations = 1000

# Aproximar valores propios usando el método de la potencia
eigenvalues_power, eigenvectors_power = power_iteration(B, num_iterations)

# Aproximar valores propios usando el método de la potencia inversa con desplazamientos
shifts = [0, -10, -20]  # Puedes ajustar los desplazamientos según sea necesario
eigenvalues_inverse_power = []
eigenvectors_inverse_power = []

for shift in shifts:
    eigenvalues, eigenvectors = inverse_power_iteration(B, shift, num_iterations)
    eigenvalues_inverse_power.append(eigenvalues)
    eigenvectors_inverse_power.append(eigenvectors)

# Imprimir los resultados
for i, eigenvalue in enumerate(eigenvalues_power):
    print(f"Valor propio aproximado {i + 1} (método de la potencia): {eigenvalue}")

for j, shift in enumerate(shifts):
    for i, eigenvalue in enumerate(eigenvalues_inverse_power[j]):
        print(f"Valor propio aproximado {i + 1} (método de la potencia inversa con desplazamiento μ={shift}): {eigenvalue}")
```

### Problema 2.b: Verifique sus resultados usando eleig Función en SciPy.
## Solucion 2.b 
```python
import numpy as np
from scipy.linalg import eig

# Define la matriz B
B = np.array([[-2, -18, 6],
              [-11, 3, 11],
              [-27, 15, 31]])

# Calcula los valores y vectores propios utilizando eig
eigenvalues, eigenvectors = eig(B)

# Imprime los resultados
print("Valores propios calculados con eig:")
for i, eigenvalue in enumerate(eigenvalues):
    print(f"Valor propio {i + 1}: {eigenvalue.real}")

```

```python
```

## PROBLEMA 2
1. Implemente un algoritmo para calcular la factorización QR de una
matríz basando en el proceso de ortogonalización de Grahm-Schmidt.
El algoritmo debe recibir una matriz A de tamaño m × n con m ≥ n
y retornar una matriz Q de tamaño m × n y una matriz triangular
superior R de tamaño n × n, tales que QtQ = In y A = QR. Compare
los resultados de su algoritmo con los de la función scipy.linalg.qr -
SciPy Manual.
2. ¿Que pasa con la factorización QR cuando las columnas son linealmente
dependientes?
3. Averigüe bajo cuales condiciones la factorización QR es única.


## PROBLEMA 3
1. Realice el siguiente tutorial sobre Topic modeling with NMF and SVD. https://nbviewer.org/github/fastai/numerical-linear-algebra/blob/master/nbs/2.%20Topic%20Modeling%20with%20NMF%20and%20SVD.ipynb
3. Realice entre 3 y 8 consultas sobre temas distintos en el portal CREA
| Real Academia Española, construya una matriz de frecuencia de términos a partir de esas consultas.
4. Realice un análisis de tópicos usando una factorización no negativa
(NMF) de la matriz construida en el punto anterior.

## PROBLEMA 4
Obtenga la descomposición en valores singulares de una foto suya en escala
de grises. Representela de nuevo utilizando sólo el valor singular mayor,
luego los dos mayores, luego los tres mayores y así hasta agotarlos. ¿Cuál
cree usted que sería en esta foto un corte óptimo?

### Solucion 4

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Cargar la imagen en escala de grises
imagen = cv2.imread('.\\h1.jpg', cv2.IMREAD_GRAYSCALE)
# Realizar la descomposición en valores singulares (SVD)
U, S, Vt = np.linalg.svd(imagen, full_matrices=False)

# Número de valores singulares a utilizar
num_valores_singulares = np.arange(1, min(imagen.shape), 10)

# Almacenar las imágenes reconstruidas
imagenes_reconstruidas = []

for k in num_valores_singulares:
    reconstruccion = np.dot(U[:, :k], np.dot(np.diag(S[:k]), Vt[:k, :]))
    imagenes_reconstruidas.append(reconstruccion)

# Visualizar las imágenes con diferentes cantidades de valores singulares
plt.figure(figsize=(15, 8))
plt.subplot(2, 5, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Original')

for i in range(9):
    plt.subplot(2, 5, i + 2)
    plt.imshow(imagenes_reconstruidas[i], cmap='gray')
    plt.title(f'{num_valores_singulares[i]} SV')

plt.show()

```

