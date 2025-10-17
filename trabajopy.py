#Se importan las librerías necesarias
import numpy as np #Para arreglos
import matplotlib.pyplot as plt #Para la parte gráfica
from matplotlib.animation import FuncAnimation #Para las animaciones


def julia_frame_iter(Z0, c, maxiter = 100, escape_radius = 4.0):
    """
    La función está diseñada para calcular el conjunto de Julia para varias condiciones
    iniciales o píxeles de la imagen. Simultáneamente, en lugar de ir punto por punto. 
    Esto es mucho más rápido en Python gracias a las optimizaciones internas de NumPy.

    Parámetros:
        Z0 : matriz de números complejos (varias condiciones iniciales, representa todos los píxeles)
        c  : constante compleja fija que define el conjunto de Julia
        maxiter : número máximo de iteraciones
        escape_radius : umbral de escape (|z|^2 > escape_radius)
    """

    # Copiamos Z0 porque lo vamos a modificar en el proceso
    Z = Z0.copy()

    # Una matriz del mismo tamaño que Z para guardar el número de 
    # iteración en la que escapó cada punto. Empieza en cero.
    iters = np.zeros(Z.shape, dtype=np.int32)

    # Creamos una "máscara booleana": True = aún no escapó
    mask = np.ones(Z.shape, dtype=bool)

    # Iteramos hasta el máximo
    for k in range(1, maxiter + 1):
        # Se realiza la operación de la recurrencia, pero usando las matrices
        # Es decir, la operación Z[mask] seleciona solo los elementos de Z cuyo
        # valor correspondiente en mask siga siendo True (es decir, los puntos que aún no han escapado)
        # Esto hace la operación más eficiente, mediante la optimización de NumPy

        Z[mask] = Z[mask] * Z[mask] + c

        # Calculamos |z|^2 usando partes reales e imaginarias (más rápido que abs())
        # Se toma el radio de escape 4.0 ya que |Z|² > (2)², es decir, es equivalente pero sin
        # tener que calcular raíces cuadradas (computacionalmente costoso)

        # Se calcula el módulo al cuadrado de todos los valores y aquellos cuyo
        # módulo supere al umbral, quedan marcados como True en la matriz booleana escaped_now
        # (se corresponden las posiciones)
        escaped_now = (Z.real * Z.real + Z.imag * Z.imag) > escape_radius


        # newly es otra matriz que aplica la operación AND entre las matrices 
        # escaped_now y mask, el objetivo es saber los nuevos puntos que acaban de superar
        # el umbral, recuerde que si un punto aún no había escapado, estaba marcado como True en mask
        # si escapa en la iteración k, se marca en escaped_now True, luego la conjunción
        # True and True es igual a True, implicando que salió en esa iteración
        # Detectamos los puntos que acaban de escapar
        newly = escaped_now & mask

        # Esta operación selecciona las entradas de newly marcadas como True y, respectivamente asigna a las
        # entradas de iters el número de la iteración donde sucedió.
        iters[newly] = k

        # Actualizamos la máscara: los que escaparon ya no se actualizan
        # Se aplica la negación a todos los elementos de newly, por lo que los puntos
        # que escaparon quedan con la máscara False
        # Luego realiza la conjunción entre mask y newly, de manera tal que los puntos que ante estaban
        # activos (marcados como True) pasan a ser Falsos en caso de que estén inactivos.
        mask &= ~newly

        # Si todos escaparon, salimos del bucle
        if not mask.any():
            break

    # A los que nunca escaparon, les ponemos el valor k máximo (asumimos que convergieron)
    iters[mask] = maxiter

    return iters



###CREACIÓN DE LA MALLA (EL PLANO COMPLEJO)

# Definimos los límites del plano complejo que vamos a visualizar
x_min, x_max = -2.0, 2.0   # Eje real
y_min, y_max = -2.0, 2.0   # Eje imaginario

# Resolución de la imagen (cuántos puntos por unidad)
density = 120

# Creamos los vectores de coordenadas reales e imaginarias
# Generan vectores con números equidistantes entre x_min y x_max
x = np.linspace(x_min, x_max, int((x_max - x_min) * density))
y = np.linspace(y_min, y_max, int((y_max - y_min) * density))

# Se toman los vectores de los ejes y se crean dos matrices, que funcionarán para la malla
# La matriz X toma el vector x y lo repite en varias filas
# La matriz Y toma el vector y, y lo repite en varias columnas
# Cada par (X[i,j]; Y[i,j]) es una coordenada del plano complejo 
X, Y = np.meshgrid(x, y)

# Combinamos las coordenadas reales e imaginarias para formar números complejos
# j representa la unidad imaginaria (i) en python. Ahora se tiene la matriz de complejos
Z0 = X + 1j * Y

# Mostramos la forma de la malla
print("Tamaño de la malla:", Z0.shape)



# ----------------------------------------------------------------------
# CONFIGURACIÓN INICIAL DE MATPLOTLIB PARA LA ANIMACIÓN
# ----------------------------------------------------------------------

# Creamos la figura (ventana) y los ejes
fig, ax = plt.subplots(figsize=(6, 6))

# Calculamos el primer frame para inicializar la imagen.
# Usamos un ángulo de 0 radianes (frame=0).
initial_c = 0.7885 * np.exp(1j * 0) #Genera el npumero complejo 0.7885 + 0i
initial_iters = julia_frame_iter(Z0, initial_c) #Calcula el conjunto de Julia para ese valor inicial

# 'imshow' devuelve un objeto Image. Este objeto es el que actualizaremos
# en cada frame con 'set_data()'.
img = ax.imshow(initial_iters, 
                extent=(x_min, x_max, y_min, y_max),
                origin='lower', 
                cmap='inferno', 
                interpolation='bicubic')

# Configuramos la barra de color y el título inicial
cbar = fig.colorbar(img, ax=ax, label='Iteraciones hasta escapar')
ax.set_title(f"c = {initial_c.real:.3f} + {initial_c.imag:.3f}i")


# ----------------------------------------------------------------------
# 4. FUNCIÓN DE ACTUALIZACIÓN
# ----------------------------------------------------------------------


def update(frame):
    # 'frame' va de 0 a 99 (total de 100 frames)
    # Genera un ciclo completo (2*pi) en 100 pasos
    angle = frame * 2 * np.pi / 100 
    
    # La constante c se mueve en un círculo de radio 0.7885
    c = 0.7885 * np.exp(1j * angle) 
    
    # Cálculo rápido del conjunto de Julia para el nuevo c
    iters = julia_frame_iter(Z0, c) 

    # Actualiza los datos de la imagen
    img.set_data(iters)
    
    # Actualiza el título
    ax.set_title(f"c = {c.real:.3f} + {c.imag:.3f}i")
    
    # Se debe devolver una tupla con los objetos que han sido modificados
    return [img]

# ----------------------------------------------------------------------
# 5. CREACIÓN Y EJECUCIÓN DE LA ANIMACIÓN
# ----------------------------------------------------------------------

# frames=100: Número de cuadros a generar (de 0 a 99)
# interval=100: Retardo entre cuadros en milisegundos (100ms = 10 cuadros por segundo)
# blit=True: Optimización para actualizar solo lo que cambia (solo la imagen)
anim = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# Muestra la ventana con la animación
# plt.show()
anim.save('julia_rapido.gif', writer='pillow', fps=20)
