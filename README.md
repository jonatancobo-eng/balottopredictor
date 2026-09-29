# Baloto Predictor Analytics

Aplicación Streamlit para analizar el histórico público de Baloto y generar combinaciones mediante un modelo heurístico ponderado.

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicar gratis en Streamlit Community Cloud

1. Crea un repositorio público en GitHub.
2. Sube `app.py` y `requirements.txt`.
3. Entra a Streamlit Community Cloud.
4. Selecciona el repositorio, rama y archivo `app.py`.
5. Pulsa Deploy.

La app intenta leer el histórico desde el sitio oficial de Baloto. Si el sitio bloquea temporalmente las solicitudes, usa un pequeño conjunto de respaldo para que la interfaz siga funcionando.

## Importante

Los sorteos de lotería están diseñados para ser aleatorios. Las frecuencias, atrasos y modelos estadísticos no garantizan que una combinación tenga mayor probabilidad real de ganar. El "Score" de la aplicación es una métrica interna del modelo.
