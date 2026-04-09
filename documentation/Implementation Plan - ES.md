# Plan de Implementación del Plugin ARTalk para LiveKit

Este documento describe la implementación actual del plugin de ARTalk para LiveKit, el cual integra el SDK externo de ARTalk para la generación de avatares 3D en tiempo real (Gaussian Splatting) con la infraestructura de comunicación en tiempo real de LiveKit.

## Componentes Principales

### 1. `avatar.py`: Coordinación de Sesión e Intercepción de Audio
- **`ARTalkAvatarSession`**: La clase principal responsable de coordinar todo el flujo de trabajo. Administra `ARTalkSDKWrapper` y `ARTalkVideoSource`.
- **`TTSWrapper` y `TTSWrapperStream`**: Interceptan el audio generado por el motor TTS (Text-to-Speech). Envían el audio hacia LiveKit para que el usuario pueda escucharlo, y simultáneamente alimentan el audio PCM (remuestreado a 16kHz) al SDK de ARTalk para la animación del avatar.
- **Procesamiento de Cuadros Asíncrono**: Ejecuta una tarea asíncrona dedicada (`_process_frame_loop`) que consulta continuamente la cola segura de subprocesos (thread-safe queue) del SDK y publica los cuadros de video en LiveKit.

### 2. `video_source.py`: Publicación de Video WebRTC
- **`ARTalkVideoSource`**: Administra el proxy personalizado `rtc.VideoSource`.
- **Procesamiento de Cuadros (`publish_frame`)**: Toma arreglos 3D de numpy (cuadros) del SDK, los redimensiona si es necesario usando OpenCV (`cv2`), convierte el formato de RGB a RGBA ya que LiveKit espera el formato RGBA, y captura el cuadro en el flujo WebRTC a través de `rtc.VideoFrame`.

### 3. `artalk_sdk.py`: Wrapper del Motor Externo ARTalk
- **`ARTalkSDKWrapper`**: Actúa como una capa de abstracción sobre el repositorio clonado de ARTalk.
- **Carga Dinámica**: Modifica `sys.path` y cambia el directorio de trabajo actual para cargar los modelos internos de ARTalk (como `Renderer`, parámetros FLAME, etc.) correctamente sin modificar profundamente el repositorio original.
- **Procesamiento de Audio**: Agrega fragmentos de audio en un búfer (`process_audio_chunk`) y los procesa para generar características de renderizado neuronal implícito mediante la extracción de características de audio (por ejemplo, HuBERT).
- **Encolamiento de Cuadros**: Coloca los cuadros visuales generados en un `queue.Queue` (seguro para subprocesos) para que sean recogidos por `ARTalkAvatarSession`.
- **Gestión de Estilos**: Permite la actualización dinámica de los estilos de habla (`update_style`) utilizando embeddings de PyTorch.
