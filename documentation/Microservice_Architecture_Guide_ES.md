# Guía de Arquitectura Microservicio para ARTalk

Esta guía documenta la nueva arquitectura implementada para desacoplar completamente ARTalk del flujo principal del Agente de LiveKit, imitando a nivel técnico la arquitectura que utilizan plataformas externas como Tavus.

---

## 🏗️ 1. Visión General de la Arquitectura

Para mantener un rendimiento óptimo de los Agentes de voz ("Cerebro") y poder escalar la pesada carga de renderizado 3D de Avatares ("Cuerpo"), el sistema se ha dividido en dos mundos completamente paralelos que se comunican de forma asíncrona a través de la API y WebRTC (LiveKit).

1. **El Servidor ARTalk (El Músculo - GPU):** Un servidor de FastAPI independiente (`artalk_server/`) que puede correr en una máquina dedicada con GPUs potentes.
2. **El Plugin Cliente (El Cerebro - CPU):** Un plugin extremadamente ligero en `livekit/plugins/artalk/` que solo despacha peticiones HTTP y envía audio, pero nunca procesa ni un solo frame de video.

---

## ⚙️ 2. El Servidor FastAPI (`artalk_server`)

### ¿Qué hace?
Es el equivalente a los servidores privados de Tavus. Contiene tres archivos principales:
- `main.py`: Expone los endpoints REST.
- `models.py`: Modelos de Pydantic para validar entradas y salidas.
- `worker.py`: La magia ocurre aquí. Es un **Cliente WebRTC Headless**. Se conecta a la sala de LiveKit, escucha el audio inyectado y publica el video resultante usando el viejo wrapper `ARTalkSDKWrapper`.

### ¿Cómo arrancarlo?
Puedes utilizar el script de conveniencia que hemos creado en `examples/start_artalk_server.py`. Este script lee las variables de entorno (`ARTALK_SERVER_HOST`, `ARTALK_SERVER_PORT` y las credenciales de LiveKit) e inicia el servidor.

Idealmente en tu máquina con GPU ("Worker node"):

```bash
cd E:\PROJECTS\PROJECT_BRAIN-AIX_VANCOUVER\SOURCE\livekit-plugins-artalk\
pip install fastapi uvicorn pydantic

# Arrancar usando el script de ejemplo programático:
python examples/start_artalk_server.py
```

### Endpoints
- **`POST /v1/avatar/create`**: Recibe una URL de imagen, usa el procesador legacy (`prepare_artalk_avatar.py`), devuelve el `replica_id` generado y la calidad (PSNR/SSIM).
- **`POST /v1/conversation`**: Recibe el `replica_id` y las credenciales del LiveKit Room. Lanza el worker en background para inyectar el video en la sala.

---

## 🔌 3. El Plugin Cliente (`artalk`)

Esta es la nueva biblioteca limpia que importas en el código de tu Asistente de Voz (Voice Agent).

### ¿Dónde vive?
`livekit-plugins-artalk/livekit/plugins/artalk/` (Para no ensuciar/eliminar tu antiguo plugin `artalk`).

### Archivos Principales
- `api.py`: Contiene `ARTalkAPI`, un cliente HTTP (`aiohttp`) que se comunica con tu servidor en el puerto 8000.
- `avatar.py`: Contiene `AvatarSession`. Intercepta el Audio Output de tu Agente para que en vez de salir "al aire" (donde provocaría eco e interferencia), se envíe directamente al Worker de la GPU usando los DataChannels (`DataStreamAudioOutput`).

### Ejemplos de Uso (Carpeta `examples/`)

Hemos creado varios scripts para facilitar el uso y las pruebas de la arquitectura:

1. **`examples/create_avatar_replica.py`**: Este script se conecta únicamente a tu servidor GPU para procesar una imagen y generar el Avatar (`preprocess_avatar_image`), devolviéndote inmediatamente el ID del Avatar y sus métricas de calidad sin iniciar una llamada de LiveKit.
2. **`examples/example_microservice_agent.py`**: Script completo que toma un Voice Agent (tu cerebro), crea el avatar remotamente, pide que el servidor GPU se conecte a la sala LiveKit actual como participante virtual, e inyecta el audio hacia el avatar usando la API `AvatarSession.start()`.

Puedes probar la integración ejecutando este último archivo:

```python
import asyncio
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.plugins.artalk import ARTalkAPI, AvatarSession

async def entrypoint(ctx: JobContext):
    # 1. Crear el avatar (Llama a tu propio servidor FastAPI por HTTP)
    api = ARTalkAPI(api_url="http://localhost:8000") # o la IP de tu Server GPU
    
    print("Creando réplica en el Cloud ARTalk...")
    replica = await api.create_replica("https://foto.com/avatar.png")
    print(f"ID del Avatar: {replica.replica_id}")

    # 2. Arrancar la Sesión
    avatar = AvatarSession(replica_id=replica.replica_id, api_url="http://localhost:8000")
    
    # Esto une el Backend GPU a tu sala usando un JWT especial y engancha el audio
    # mediante Data Channels, exactamente igual que el plugin de Tavus.
    await avatar.start(ctx.agent, ctx.room)
    
    # Listo. Tu agente de voz puede seguir usando OpenAI, Cartesia, etc...
```

> [!IMPORTANT]
> **Sobre el paso de la Sesión al Avatar**: Cuando llamas a `avatar.start(agent_session=session, ...)`, es crítico que pases el objeto de tu Pipeline o Cerebro (ej. `AgentSession` o `VoicePipelineAgent`) y **NO** el objeto `ctx.agent` (que es el `LocalParticipant`). El avatar necesita secuestrar el audio extrayéndolo directamente del generador TTS de tu Pipeline antes de salir a la red.
>
> Asimismo, al arrancar tu sesión final con `session.start()`, debes pasar obligatoriamente una instancia de una subclase tuya (ej. `agent=TuAgentePersonalizado()`) junto con los parámetros nombrados (kwargs), en lugar de la clase base vacía `Agent()`, para evitar errores (Overloads) del editor de código como Pylance.

---

## ✅ Resumen de la Separación de Lógicas

| Módulo | Antes | Ahora (Arquitectura Microservicio) |
| :--- | :--- | :--- |
| **Generación FLAME / GAGAvatar** | Bloqueaba el hilo del Agente de Voz (GIL). | Corre aislado en `artalk_server/worker.py` en otro proceso/máquina. |
| **Envío de Video** | El Agente de Voz publicaba la pista de video iterando frames. | El Backend de API (`worker.py`) se une a la sala como participante virtual y publica el video directamente al SFU de LiveKit. |
| **Sincronización Audio-Video** | Se hacía a la fuerza empatando contadores locales. | El Agente de Voz transmite bytes de audio por *DataChannel* a `worker.py`, que los encola y procesa asíncronamente con su video. |
| **`prepare_artalk_avatar.py`** | Script de CLI manual local. | Se importó y wrappeó en un Endpoint HTTP (`POST /v1/avatar/create`). |
