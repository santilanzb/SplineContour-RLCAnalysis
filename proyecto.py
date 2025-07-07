import cv2
import numpy as np

imagen = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if imagen is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el nombre del archivo.")
else:
    # Aplicar el detector de bordes de Canny
    bordes = cv2.Canny(imagen, 100, 200)

    # Mostrar la imagen original en escala de grises
    cv2.imshow('Imagen Original en Escala de Grises', imagen)

    # Mostrar los bordes detectados
    cv2.imshow('Bordes Detectados con Canny', bordes)

    # Tocar cualquier tecla para salir
    cv2.waitKey(0)

    # Cerrar todas las ventanas de OpenCV
    cv2.destroyAllWindows()

    # Guardar la imagen con los bordes detectados
    cv2.imwrite('bordes_panda_canny.jpg', bordes)
    print("Los bordes han sido detectados y guardados como 'bordes_panda_canny.jpg'")