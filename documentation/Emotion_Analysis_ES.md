# Análisis de Emociones del Usuario en LiveKit

Para lograr el reconocimiento de emociones casi en tiempo real a partir del video y el audio de un usuario usando LiveKit, la mejor estrategia es aprovechar el **LiveKit Agents Framework** (o un cliente backend de LiveKit).

## Cómo funciona el sistema

LiveKit maneja de forma nativa el complejo protocolo WebRTC, pero para analizar el video y el audio con Inteligencia Artificial, necesitamos capturar los datos "crudos" (raw data). Este es el flujo de trabajo paso a paso:

1.  **Suscripción a las Pistas (Tracks):** Tu aplicación backend se conecta a la sala y reacciona al evento `track_subscribed` cuando el usuario enciende su cámara y micrófono.
2.  **Creación del Flujo (Stream):** Una vez suscrito, inicializas un `rtc.VideoStream(track)` para el video y un `rtc.AudioStream(track)` para el audio. Estos actúan como bucles asíncronos que entregan los fotogramas (frames) crudos.
3.  **Muestreo (Downsampling):** Procesar video a 30 FPS y audio continuo es muy pesado e innecesario para detectar emociones. El sistema debe muestrear los datos. Por ejemplo, capturar solo 1 fotograma de video por segundo, y agrupar el audio en fragmentos (chunks) de 1 o 2 segundos.
4.  **Inferencia (Procesamiento de IA):**
    *   **Video:** El `rtc.VideoFrame` se convierte en una imagen convencional (como un array `numpy` de OpenCV o formato RGB) y se pasa a un modelo de Visión Computacional para detectar *Expresiones Faciales*.
    *   **Audio:** Los fragmentos crudos de audio PCM se envían a un modelo de Reconocimiento de Emociones del Habla (SER, por sus siglas en inglés) para analizar el tono de voz.
5.  **Acción/Salida:** La emoción detectada (ej. "Feliz: 80%, Surprised: 20%") se envía casi en tiempo real de vuelta al frontend usando los **Canales de Datos de LiveKit** (`room.local_participant.publish_data()`), o bien se utiliza internamente para cambiar el comportamiento de un Avatar de IA.

## Opciones para implementarlo

### 1. Hume AI (Recomendado para IA Emocional en Tiempo Real)
Hume AI se especializa en "Medición de Expresiones". Ofrecen APIs en streaming (mediante WebSockets) que pueden recibir tanto audio como video al mismo tiempo para medir emociones humanas de forma perfecta en tiempo real. Es la opción comercial más robusta y precisa ahora mismo.

### 2. APIs en la Nube Estándar (AWS Rekognition / Google Cloud / Azure)
Puedes enviar tus fotogramas a AWS Rekognition (que tiene una API `DetectFaces` que devuelve emociones) y el audio a las APIs semánticas de Google/Azure. Es excelente para escalar, aunque puede introducir una ligera latencia de red.

### 3. Modelos Locales de Código Abierto (Ideal para Privacidad y Cero Costo)
*   **Para Video:** Puedes correr el modelo **DeepFace** (Python) de forma local en tu backend. Es una librería muy ligera que puede analizar emociones (enojo, miedo, neutral, tristeza, asco, alegría, sorpresa) desde un solo fotograma usando OpenCV.
*   **Para Audio:** Puedes usar modelos gratuitos de Hugging Face como los de **SpeechBrain** para clasificar la emoción a partir de los fragmentos de audio de 1 segundo.
