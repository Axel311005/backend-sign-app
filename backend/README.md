# Señas y Sonrisas API

Backend Flask que expone el modelo de letras y numeros para inferencia en tiempo real.

## Requisitos

- Python 3.10+
- Modelos en `../Code/` (`modelo_2209_miguel.h5`, `modelmiguelnumber.h5`)

## Instalacion

```bash
cd backend
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Nota: los modelos .h5 fueron entrenados con Keras legacy. Este backend usa
`TF_USE_LEGACY_KERAS=1` y `tf-keras` para cargarlos correctamente.

## Ejecutar

```bash
python app.py
```

## Endpoints

- `GET /health`
- `POST /predict`

### Ejemplo de payload

```json
{
  "mode": "letters",
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

### Respuesta

```json
{
  "mode": "letters",
  "label": "A",
  "score": 0.92,
  "top3": [
    { "label": "A", "score": 0.92 },
    { "label": "S", "score": 0.04 },
    { "label": "E", "score": 0.02 }
  ],
  "latency_ms": 18
}
```
