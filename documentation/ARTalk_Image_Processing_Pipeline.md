# ARTalk Image Processing Pipeline / Pipeline de Procesamiento de Imágenes de ARTalk

*(Versión en Español abajo / Spanish version below)*

---

## 🇺🇸 English: How ARTalk Processes Avatar Images

ARTalk does not process raw 2D images (like `.jpg` or `.png`) "on-the-fly" during real-time video inference. Instead, the visual rendering relies on an auxiliary model called **GAGAvatar** (Generalizable and Animatable Gaussian Head Avatar). 

The system uses a key variable called **`shape_id`** to identify which 3D model to animate. When a user selects an image, ARTalk doesn't render the image directly; it queries a pre-computed 3D structure database using that image's filename as the ID.

### The 4-Step Technical Pipeline

#### 1. Image Retrieval (Obtaining the Image)
The pipeline starts by receiving the physical path to a raw image (e.g., `/path/to/my_photo.jpg`). In our LiveKit integration, this is passed dynamically via an environment variable (`AVATAR_IMAGE`). 
- **Requirement:** A valid frontal facial image in `.jpg`, `.jpeg`, or `.png` format.

#### 1b. Image Validation (The Service Check)
Before tracking, the microservice runs three mandatory sanity checks:
1. **Resolution:** Minimum 256×256 pixels.
2. **Quality (Sharpness):** Detects blur or near-blank images.
3. **Face Detection:** Confirms that a human face is identifiable.
*If any validation fails, the service reports a descriptive error in English and aborts the pipeline.*

#### 2. GAGAvatar Preprocessing (The Pre-flight Track)
Before the ARTalk agent starts rendering video, the image must be converted into a mathematical representation of a 3D head.
- This is done using the `GAGAvatar_track.engines.CoreEngine` (TrackEngine).
- The engine analyzes the image to extract facial landmarks, posture, and **FLAME parameters** (a statistical 3D head model framework) alongside camera parameters.
- These extracted features are then serialized and appended as a new entry into a global PyTorch dictionary file located at `ARTalk/render_results/tracked/tracked.pt`.

#### 3. Key Generation (`shape_id`)
When the preprocessing engine saves the mathematical data into the `tracked.pt` database, it needs a unique key for retrieval.
- **The Key:** GAGAvatar uniquely identifies the processed avatar using the **exact base filename** of the input image. 
- For example, if the input is `my_photo.jpg`, the key stored in the dictionary will simply be `"my_photo.jpg"`. This string becomes the `shape_id`.

#### 4. Directed Rendering
Once the image is mathematically tracked and indexed:
- The generated `shape_id` (`"my_photo.jpg"`) is passed dynamically to the `ARTalkAvatarSession` during initialization.
- When ARTalk's main inference loop runs (driven by Cartesia TTS audio), it looks up `"my_photo.jpg"` inside the `tracked.pt` database.
- It retrieves the base FLAME structure, applies the desired motion/speaking `style_id` to deform the mesh according to the audio, and injects the user's original visual texture to render the final Gaussian Splatting video frames into the WebRTC stream.

---

## 🇪🇸 Español: Cómo procesa ARTalk las imágenes de los Avatares

ARTalk no procesa imágenes 2D crudas (como `.jpg` o `.png`) "al vuelo" durante la inferencia de video en tiempo real. En su lugar, el renderizado visual recae sobre un modelo auxiliar llamado **GAGAvatar** (Avatares de Cabeza Gaussiana Generalizables y Animables). 

El sistema utiliza una variable clave llamada **`shape_id`** para identificar qué modelo 3D debe animar. Cuando un usuario selecciona una imagen, ARTalk no la renderiza directamente; sino que consulta una base de datos de estructuras 3D precalculadas usando el nombre de archivo de esa imagen como identificador.

### El Pipeline Técnico de 4 Pasos

#### 1. Obtención de la Imagen
El proceso comienza recibiendo la ruta física de una imagen cruda (ej. `/ruta/a/mi_foto.jpg`). En nuestra integración con LiveKit, esto se pasa dinámicamente a través de una variable de entorno (`AVATAR_IMAGE`).
- **Requisito:** Una imagen facial frontal válida en formato `.jpg`, `.jpeg`, o `.png`.

#### 1b. Validación de la Imagen (Control del Servicio)
Antes del rastreo, el microservicio ejecuta tres controles de integridad obligatorios:
1. **Resolución:** Mínimo de 256×256 píxeles.
2. **Calidad (Nitidez):** Detecta imágenes borrosas o casi vacías.
3. **Detección de Rostro:** Confirma que el rostro es identificable.
*Si alguna validación falla, el servicio informa un error descriptivo en inglés y detiene el proceso.*

#### 2. Preprocesamiento con GAGAvatar (Tracking previo)
Antes de que el agente ARTalk empiece a renderizar el video, la imagen debe convertirse en una representación matemática de una cabeza 3D.
- Esto se logra utilizando el motor `GAGAvatar_track.engines.CoreEngine` (TrackEngine).
- El motor analiza la foto para extraer puntos de referencia faciales, postura y **parámetros FLAME** (un marco estadístico para modelos 3D de cabezas) junto con los parámetros de la cámara.
- Estas características extraídas se serializan y se añaden como una nueva entrada en un archivo de diccionario global de PyTorch ubicado en `ARTalk/render_results/tracked/tracked.pt`.

#### 3. Generación de Clave (`shape_id`)
Cuando el motor de preprocesamiento guarda los datos matemáticos en la base de datos `tracked.pt`, necesita una clave única para su posterior recuperación.
- **La Clave:** GAGAvatar identifica unívocamente al avatar procesado utilizando el **nombre de archivo base exacto** de la imagen de entrada. 
- Por ejemplo, si la entrada es `mi_foto.jpg`, la clave guardada en el diccionario será simplemente `"mi_foto.jpg"`. Esta cadena de texto se convierte en el famoso `shape_id`.

#### 4. Orientación de la Renderización
Una vez que la imagen ha sido rastreada matemáticamente e indexada:
- El `shape_id` generado (`"mi_foto.jpg"`) se le pasa dinámicamente a la capa `ARTalkAvatarSession` durante la inicialización de LiveKit.
- Cuando el bucle de inferencia principal de ARTalk se ejecuta (impulsado por el audio TTS de Cartesia), busca `"mi_foto.jpg"` dentro de la base de datos `tracked.pt`.
- Recupera la estructura FLAME base, aplica el estilo de movimiento/habla deseado (`style_id`) para deformar la malla según el audio, e inyecta la textura visual original del usuario para renderizar los fotogramas finales de "Gaussian Splatting" hacia el flujo de WebRTC.
