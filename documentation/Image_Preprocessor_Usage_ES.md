# Preprocesador de Avatares ARTalk (`prepare_artalk_avatar.py`)

Esta herramienta te permite procesar fácilmente una nueva imagen de avatar (ya sea un archivo local o una URL) para que pueda ser utilizada por el modelo ARTalk. Ejecuta el pipeline `GAGAvatar_track` para extraer la configuración FLAME de la imagen y guarda los resultados directamente en la base de datos `tracked.pt` de ARTalk.

## Requisitos Previos

1.  Asegúrate de haber instalado el plugin de ARTalk o sus dependencias.
2.  Debes conocer la ruta absoluta o relativa a tu instalación del repositorio ARTalk (usualmente `./external_models/ARTalk`).

## Instrucciones de Uso

Puedes ejecutar el script desde la raíz del repositorio `livekit-plugins-artalk`.

### 1. Procesar una Imagen Local

```bash
python prepare_artalk_avatar.py "ruta/a/mi_imagen.jpg" --artalk_path "./external_models/ARTalk"
```

### 2. Procesar una Imagen desde una URL

```bash
python prepare_artalk_avatar.py "https://ejemplo.com/fotografia.png" --artalk_path "./external_models/ARTalk"
```

### Argumentos del Script

*   `image_input`: (Requerido) La ruta a la imagen local (`.jpg`, `.png`) o URL (`http://` o `https://`).
*   `--artalk_path`: (Requerido) La ruta al repositorio original de ARTalk.
*   `--device`: (Opcional) El dispositivo en el que se ejecutará el rastreador (tracker) (por defecto: `cuda`).
*   `--no_matting`: (Opcional) Omitir el paso de recorte de fondo (matting). Es más rápido pero puede incluir artefactos en la salida.

## Validaciones Automáticas de Imagen

Cuando utilizas este script a través del microservicio ARTalk (`/v1/avatar/create`), se realizan varios controles automáticos **antes** de que comience el proceso de rastreo:

1.  **Control de Resolución:** La imagen debe tener al menos **256x256 píxeles**.
2.  **Control de Nitidez:** Utiliza la varianza Laplaciana para asegurar que la imagen no esté demasiado borrosa (puntuación mínima: 50.0).
3.  **Detección de Rostro:** Utiliza MediaPipe (o el fallback de OpenCV Haar) para confirmar que hay un rostro humano en la imagen.

Si alguno de estos falla, el servicio devolverá un mensaje de error detallado en inglés. Asegúrate de que las imágenes de entrada sean retratos frontales claros y bien iluminados para obtener los mejores resultados.

## Cómo Usar el Avatar Procesado

Una vez que el script termine con éxito, mostrará un **Avatar ID** generado (por ejemplo, `mi_imagen.jpg`).

### En un Agente LiveKit

Puedes usar el ID generado directamente en el código de tu agente:

```python
from livekit.plugins.artalk import ARTalkAvatarSession

# Pasa el ID impreso por el comando
avatar = ARTalkAvatarSession(
    shape_id="mi_imagen.jpg", 
    # ... otros parámetros del agente
)
```

### En la App Original de Gradio de ARTalk (`inference.py`)

Si ejecutas la interfaz web de prueba original de ARTalk:

```bash
cd external_models/ARTalk
python inference.py --run_app
```

El ID de tu imagen aparecerá automáticamente en el menú desplegable "Choose the appearance of the speaker" ("Elige la apariencia del hablante") bajo la sección "Avatar Control" ("Control de Avatar").

*   **Consejo:** Si también quieres que tu imagen aparezca como uno de los recuadros de ejemplo seleccionables en la parte inferior, edita el código fuente de `inference.py` y añade tu configuración a la lista `examples` (alrededor de la línea ~178):
    ```python
    ["Audio", "...", None, None, "mi_imagen.jpg", "natural_0"]
    ```
