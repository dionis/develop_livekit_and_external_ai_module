# Análisis de Métricas de Inferencia en ARTalk

Este documento detalla las métricas utilizadas para evaluar el modelo ARTalk y analiza la viabilidad de integrarlas en el plugin de LiveKit.

## 1. Métricas de Evaluación en ARTalk

Basado en la investigación del código fuente y el artículo técnico (*paper*) de ARTalk (2025), las métricas se dividen en dos categorías principales: precisión del movimiento y calidad de la imagen.

### A. Sincronización Labial y Movimiento Facial
Estas métricas evalúan qué tan bien coinciden los movimientos generados con el audio de entrada.

*   **LMD (Landmark Distance):** Mide la distancia euclidiana entre los puntos de referencia faciales (especialmente la boca) generados y los objetivos. Un LMD bajo indica alta precisión geométrica.
*   **F-LMD (Frame-wise LMD):** Una variante del LMD que evalúa la estabilidad del movimiento cuadro por cuadro para evitar saltos bruscos (*jitter*).
*   **SyncNet Score (Confidence/Offset):** Utiliza un modelo pre-entrenado (SyncNet) para detectar si audio y video están en fase. Proporciona una puntuación de confianza y un desfase temporal (offset) en milisegundos.

### B. Calidad Visual y Realismo (GAGAvatar)
Como ARTalk utiliza **GAGAvatar** para el renderizado basado en Gaussian Splatting, se usan métricas de visión por computadora estándar:

*   **PSNR (Peak Signal-to-Noise Ratio):** Mide la calidad de reconstrucción de la imagen. Valores más altos indican menos ruido.
*   **SSIM (Structural Similarity Index):** Evalúa la similitud de estructuras, texturas y contraste entre el fotograma generado y la imagen original.
*   **LPIPS (Learned Perceptual Image Patch Similarity):** Mide la similitud "perceptual" usando redes neuronales profundas. Se considera más cercana al juicio humano que PSNR/SSIM.
*   **FID (Fréchet Inception Distance):** Evalúa el realismo general comparando la distribución de imágenes generadas con un conjunto de imágenes reales.

---

## 2. Posibilidad de Integración en este Proyecto

### Viabilidad Técnica
La integración es **factible** ya que el entorno actual utiliza PyTorch y cuenta con la mayoría de las dependencias base.

### Propuesta de Implementación

> [!IMPORTANT]  
> No se recomienda ejecutar estas métricas en tiempo real durante una sesión de LiveKit, ya que el cálculo de LPIPS o SyncNet añadiría una latencia significativa que degradaría la experiencia del usuario.

| Caso de Uso | Recomendación |
| :--- | :--- |
| **Control de Calidad (Offline)** | **Altamente recomendado.** Crear un script de validación que procese un video de prueba y reporte las métricas antes de "aprobar" un nuevo avatar. |
| **Monitoreo en Tiempo Real** | **No recomendado.** El costo computacional es demasiado alto para el beneficio inmediato. |
| **Ajuste de Estilo (Fine-tuning)** | **Útil.** Puede ayudar a decidir qué `style_id` funciona mejor para una voz específica. |

### Dependencias Necesarias
Para integrar estas métricas, deberíamos añadir:
*   `lpips` (para similitud perceptual).
*   `scikit-image` (para SSIM/PSNR simplificado).
*   `syncnet-python` (opcional, para validación rigurosa de lip-sync).

---

## 3. Próximos Pasos Sugeridos

1.  **Implementar un Módulo de Evaluación:** Crear un archivo `livekit/plugins/artalk/evaluation.py` que permita calcular estas métricas de forma asíncrona.
2.  **Benchmark de Avatares:** Usar este módulo para generar un reporte de calidad cada vez que se procesa una nueva imagen con `prepare_artalk_avatar.py`.
