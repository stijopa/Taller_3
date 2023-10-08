# Taller 3 - Factorización de matrices
NOTA: Lea sobre el método de potencias para aproximar el valor propio mayor de
una matríz. Realice los dos ejercicios que se encuentran al final de esta guía:
Approximating Eigenvalues  -  Jupyter Guide to Linear Algebra.


Problema 2
1. Implemente un algoritmo para calcular la factorización QR de una
matríz basando en el proceso de ortogonalización de Grahm-Schmidt.
El algoritmo debe recibir una matriz A de tamaño m × n con m ≥ n
y retornar una matriz Q de tamaño m × n y una matriz triangular
superior R de tamaño n × n, tales que QtQ = In y A = QR. Compare
los resultados de su algoritmo con los de la función scipy.linalg.qr 
SciPy Manual.
2. ¿Que pasa con la factorización QR cuando las columnas son linealmente
dependientes?
3. Averigüe bajo cuales condiciones la factorización QR es única.
