# ARTalk Visual Plugin for LiveKit

Este documento describe la propuesta de implementación de un plugin visual en LiveKit para integrar el modelo **ARTalk** (generador automático de animación de cabeza 3D manejado por audio).
El diseño se basa en la estructura del plugin de ejemplo `livekit-plugins-ditto`, adaptado para usar el motor de inferencia de ARTalk.

## Propuesta de Diseño Estructural

El plugin estará empaquetado como `livekit-plugins-artalk` y constará de los siguientes componentes principales:

### 1. `livekit/plugins/artalk/avatar.py`
Este archivo definirá la sesión del avatar (`ARTalkAvatarSession`) que conectará la sala de LiveKit con ARTalk.
- **TTSWrapper & TTSWrapperStream**: Sobrescriben el comportamiento de Text-to-Speech (TTS) definido en `livekit.agents`. Su objetivo es enviar el audio generado por síntesis de voz tanto a la sala de LiveKit (para que el usuario lo escuche) como al flujo de entrada de ARTalk para generar los fotogramas (lipsync).
- **`ARTalkAvatarSession`**: Controla el ciclo de vida del avatar virtual (inicio, conexión a la Room de LiveKit y finalización) y lanza una tarea asíncrona dedicada a recolectar y publicar en formato de video los fotogramas que emite ARTalk.

### 2. `livekit/plugins/artalk/artalk_sdk.py`
Envoltorio (SDK Wrapper) en torno a la implementación nativa de ARTalk, idealmente interactuando con las funciones internas de renderizado y Gaussians Splatting de su inferencia.
- **`ARTalkSDKWrapper`**: Cargará el modelo alojado de ARTalk (haciendo uso de `torchaudio`, `torch`, y el rasterizador gaussiano `diff-gaussian-rasterization`).
- Proveerá métodos como `process_audio_chunk(audio_bytes)` para enviar ráfagas de audio PCM flotante y `get_frame_queue()` para consumir los *frames* renderizados por ARTalk (arreglos HxWxC RGB o RGBA).

### 3. `livekit/plugins/artalk/video_source.py`
Manejador de video que traduce las matrices (arrays) numéricos a un flujo compatible con WebRTC usando el SDK de LiveKit.
- Utiliza `rtc.VideoSource` y `rtc.LocalVideoTrack.create_video_track` para crear la pista de video que se publicará en la `Room`.
- Contiene el bucle `publish_frame` que se encarga de cambiar formatos, redimensionar (si dictado por ARTalk) y llamar a `capture_frame()`.

## Componentes y Cambios Propuestos

### `pyproject.toml` (Propuesto)

```toml
[project]
name = "livekit-plugins-artalk"
version = "0.1.0"
description = "LiveKit Agents plugin for ARTalk 3D TalkingHead avatar integration"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "livekit~=0.17",
    "livekit-agents~=0.12",
    "numpy",
    "opencv-python-headless",
    "scipy",
    "soundfile"
]

[project.optional-dependencies]
gpu = [
    "torch>=2.1.0",
    "torchaudio>=2.1.0",
    "diff-gaussian-rasterization" # Extension CUDA propia de ARTalk
]
```

### Fragmentos Principales de Código (Propuestos)

#### `artalk_sdk.py`
```python
import numpy as np
import queue

class ARTalkSDKWrapper:
    def __init__(self, artalk_path: str, shape_id: str = "mesh", style_id: str = "default"):
        self.artalk_path = artalk_path
        self.shape_id = shape_id
        self.style_id = style_id
        self._is_loaded = False
        self.frame_queue = queue.Queue(maxsize=30)
        
    def load(self):
        # Aquí importarías dinámicamente o añadirías de sys.path el módulo ARTalk
        # desde github.com/xg-chu/ARTalk
        # e.g., inicialización del FLAME model y renderizado Gaussiano.
        self._is_loaded = True
        
    def process_audio_chunk(self, audio_data: np.ndarray):
        # Procesamiento de audio entrante -> Generar fotogramas 3D usando ARTalk
        # Ejemplo conceptual:
        # frames = self._artalk_model.infer(audio_data)
        # for frame in frames:
        #     self.frame_queue.put(frame)
        pass

    def get_frame_queue(self):
        return self.frame_queue
```

#### `avatar.py`
```python
from livekit import rtc, agents
from .artalk_sdk import ARTalkSDKWrapper
from .video_source import ARTalkVideoSource

class ARTalkAvatarSession:
    def __init__(self, artalk_path: str):
        self.sdk = ARTalkSDKWrapper(artalk_path)
        self.video_source = ARTalkVideoSource(width=512, height=512, fps=25)
        self._is_running = False

    def wrap_tts(self, tts_instance: agents.tts.TTS) -> agents.tts.TTS:
        # Devuelve el TTS Wrapper que alimenta de audio a self.sdk
        return TTSWrapper(tts_instance, self.sdk)

    async def start(self, room: rtc.Room, identity: str = "artalk-avatar"):
        self.sdk.load()
        video_track = self.video_source.create_track(f"{identity}-video")
        
        await room.local_participant.publish_track(
            video_track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
        )
        self._is_running = True
        self.video_source.start_publishing()
        # Iniciar bucle de extracción de fotogramas asíncrono
        # asynio.create_task(self._process_frames())
```

## Plan de Verificación
### Verificación Manual
1. Construir un agente tipo "Echo" (`basic_agent.py`) usando el plugin ARTalk.
2. Pasar `ARTalkAvatarSession` al configurar el TTS del agente.
3. Conectarse a LiveKit Sandbox y verificar que al hablar con el agente, este no solo contesta voz desde TTS sino que el track de Video procesa labios y gesticulaciones desde ARTalk en tiempo real de acuerdo a su archivo de entrada.

## Resolución de Problemas (Troubleshooting)

### 1. Errores de Dependencia FFmpeg (`libavfilter.so.9` / `libavutil.so.57`)
ARTalk depende en gran medida de `torchaudio` para el análisis y el remuestreo de audio. En los entornos Conda aislados, las versiones recientes de `torchaudio` pueden requerir bibliotecas antiguas de FFmpeg que el sistema no posee.
**Solución**: Asegúrate de que `ffmpeg<7` esté instalado en el entorno conda a través de `conda-forge`, en lugar de depender únicamente de la implementación `apt` del sistema, para satisfacer estos límites del enlazador dinámico. De lo contrario fallará al iniciar la interfaz del avatar.

### 2. Caída del Filtro Savitzky-Golay (`window_length must be less than or equal to the size of x`)
Al traducir trozos de audio TTS a códigos de movimiento, ARTalk aplica un filtro `scipy.signal.savgol_filter` con un `window_length=9` para suavizar las expresiones faciales y las poses de la cabeza. Si el trozo entrante es inferior a 0,36 segundos (lo que produce menos de 9 fotogramas de movimiento), el motor se bloquea.
**Solución**: El SDK utiliza un búfer de agregación interno (`self._audio_buffer`) para reenviar al motor de inferencia sólo los fragmentos mayores que un umbral (por ejemplo, 1,0s), con un corte mínimo estricto de 0,25s para descartar remanentes diminutos después de los vaciados del stream.

### 3. Fallos de Renderizado / Rayas Diagonales en el Video
Debido a que `np.dstack` y `np.transpose` producen fragmentos de memoria transpuestos, inyectar los bytes directamente a LiveKit (`frame.tobytes()`) rompe el salto (stride) de los píxeles, lo que ocasiona que la imagen se vea oblicua y con artefactos visuales graves.
**Solución**: Es obligatorio garantizar el orden con `np.ascontiguousarray(frame)` antes de inyectar las matrices RGB en el contenedor `rtc.VideoFrame`. Además, como buena práctica se enlazan los rangos de color usando `np.clip(..., 0, 255)` al re-convertir desde los tensores de StyleGAN para evitar parásitos visuales.

### 4. Interrupción de Audio y Video a "1 FPS"
Existen dos cuellos de botella arquitectónicos que bloquean la reproducción en tiempo real:
1. El motor de PyTorch de ARTalk es síncrono. Si se inserta directamente dentro del bucle asíncrono del TTS, bloquea el hilo de eventos principal, congelando el flujo de Cartesia e interrumpiendo el audio.
2. Si el generador vacía una ráfaga súbita de 25 fotogramas hacia el servidor proxy WebRTC *al mismo instante*, este último descarta los excedentes superpuestos publicando solo el último, lo que decantaba en un video torpe a ~1 fram por segundo.
**Solución**: 
- Aísla la carga pesada de inferencia en `process_audio_chunk` dentro de un enlazador `await asyncio.to_thread()`.
- Modula el bucle de publicación WebRTC matemáticamente respecto al Loop de eventos nativo de LiveKit, asegurando que los fotogramas se escurran al hilo al ritmo exacto impuesto por el formato de video (ej: `1.0 / fps_objetivo`).
