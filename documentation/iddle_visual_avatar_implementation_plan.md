# ARTalk Avatar Implementation Guide

Este documento explica el proceso técnico para mostrar el avatar, gestionar los estados de ARTalk y lograr un parpadeo natural dentro de la integración con LiveKit.

## 1. Proceso de Visualización y Movimientos Ociosos (Idle)

El proceso se divide en dos bucles principales que se alternan según si el agente está hablando o no.

### Visualización del Avatar
1. **Inicialización**: `ARTalkAvatarSession.start()` carga los modelos (SDK) y registra un `LocalVideoTrack` en la sala de LiveKit.
2. **Pipeline de Video**: Los frames (matrices NumPy) generados por el modelo se envían a [ARTalkVideoSource](file:///e:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/livekit-plugins-artalk/livekit/plugins/artalk/video_source.py#17-150), que los convierte al formato `BGRA` esperado por WebRTC y los publica en el track.
3. **Loop de Procesamiento**: [_process_frame_loop](file:///e:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/livekit-plugins-artalk/livekit/plugins/artalk/avatar.py#330-411) consume frames de una cola asíncrona y los emite a 25 fps estables para evitar variaciones en la fluidez (jitter).

### Movimientos Ociosos (Idle)
* **Mecánica**: El [_idle_loop](file:///e:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/livekit-plugins-artalk/livekit/plugins/artalk/avatar.py#412-454) se activa cuando `_is_speaking` es `False`.
* **Inyección de Silencio**: En lugar de no hacer nada, el sistema inyecta bloques de "audio silencioso" (400ms) al SDK.
* **Inferencia**: ARTalk procesa este silencio a través del modelo. El modelo de IA, al no detectar fonemas, genera micromovimientos naturales (balanceo de cabeza, respiración) en lugar de movimientos labiales.

## 2. Estados Permitidos por ARTalk

La integración gestiona los siguientes estados funcionales:

| Estado | Variable Principal | Descripción |
| :--- | :--- | :--- |
| **Reposapiés (Idle)** | `_is_speaking = False` | El bucle idle genera frames basados en silencio. |
| **Hablando (Speaking)** | `_is_speaking = True` | El SDK procesa audio real del TTS. El bucle idle se pausa automáticamente. |
| **Buffering** | `_is_buffering = True` | Se acumula un buffer de jitter (aprox. 8 frames) al inicio del habla para garantizar fluidez si la GPU tiene picos de carga. |
| **Flushing** | [flush_audio()](file:///e:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/livekit-plugins-artalk/livekit/plugins/artalk/artalk_sdk.py#392-421) | Procesa el audio residual al final de una frase para que las palabras no se corten visualmente. |

## 3. Lógica del Parpadeo Natural

El parpadeo en modelos basados en FLAME (como ARTalk) se controla mediante coeficientes de expresión específicos.

### Cómo se logra:
1.  **Identificación**: En el modelo FLAME, el parpadeo (cierre de párpados) se controla mediante los coeficientes de expresión **19** (ojo izquierdo) y **20** (ojo derecho).
2.  **Inyección Periódica**: En [artalk_sdk.py](file:///e:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/livekit-plugins-artalk/livekit/plugins/artalk/artalk_sdk.py), inyectamos manualmente valores en estos índices cada 3-6 segundos para simular el parpadeo natural.
3.  **Habilitación de Parámetros**: Se ha verificado que la línea `pred_motions[..., 104:] *= 0.0` no bloquea estos índices (0-99), por lo que la inyección es directa sobre el tensor de movimientos predichos.

## Verificación Sugerida

### Pruebas Manuales
1. **Idle Check**: Observar al avatar sin hablar. Debería tener movimientos sutiles.
2. **Latency Check**: Verificar que el avatar empiece a hablar inmediatamente después de que el TTS genere audio (el [clear_idle_state](file:///e:/PROJECTS/PROJECT_BRAIN-AIX_VANCOUVER/SOURCE/livekit-plugins-artalk/livekit/plugins/artalk/avatar.py#231-251) purga los frames de silencio acumulados).
3. **Blink Check**: Verificar si los ojos se cierran periódicamente.
