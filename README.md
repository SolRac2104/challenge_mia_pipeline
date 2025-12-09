# challenge_mia_pipeline

Challence_mia_pipeline para demostración de habilidades técnicas

## Módulo de reconocimiento facial

El repositorio incluye `facial_recognition.py`, un módulo sencillo basado en
OpenCV (LBPH) para entrenar un modelo con carpetas de imágenes y reconocer
rostros en nuevas fotografías.

### Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Preparar el dataset

Organiza tus datos como carpetas por persona:

```
dataset/
├── ana/
│   ├── foto1.jpg
│   └── foto2.jpg
└── juan/
    ├── img1.png
    └── img2.png
```

El módulo detectará rostros en cada imagen antes de entrenar.

### Uso básico

```python
from facial_recognition import FaceRecognizer

recognizer = FaceRecognizer()
recognizer.train("./dataset")
resultados = recognizer.recognize("./foto_prueba.jpg")
print(resultados)

# Guardar una copia anotada
recognizer.annotate("./foto_prueba.jpg", "./salida/anotada.jpg")
```

`recognize` devuelve una lista de `Recognition` con la etiqueta, la confianza y
el bounding box detectado. `annotate` escribe una imagen con rectángulos y la
etiqueta predicha para cada rostro.
