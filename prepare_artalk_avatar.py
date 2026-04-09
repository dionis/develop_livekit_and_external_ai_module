import os
import sys
import argparse
import urllib.request
import tempfile
from pathlib import Path

# Try to import the ARTalk plugin from the current directory if running from source
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from livekit.plugins.artalk.image_preprocessor import preprocess_avatar_image
    from livekit.plugins.artalk.evaluation import evaluate_avatar_quality, display_metrics
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("Error: Could not import livekit.plugins.artalk modules. Ensure you are running this from the project root or the plugin is installed.")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def download_image(url: str, dest_path: str):
    """Downloads an image from a URL to the specified destination path."""
    logger.info(f"Downloading image from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with open(dest_path, 'wb') as f:
            f.write(response.read())
    logger.info("Download complete.")

def main():
    parser = argparse.ArgumentParser(description="Preprocesar una imagen o URL para generar los archivos ('cocinar' el avatar) que usa el modelo ARTalk.")
    parser.add_argument("image_input", help="Ruta local a la imagen (.jpg, .png) o una URL (http:// o https://).")
    parser.add_argument("--artalk_path", required=True, help="Ruta absoluta al repositorio de ARTalk original (ej. ./external_models/ARTalk).")
    parser.add_argument("--device", default="cuda", help="Dispositivo en el que correrá el tracker (por defecto: cuda).")
    parser.add_argument("--no_matting", action="store_true", help="Omitir el recorte del fondo (matting). Más rápido pero puede incluir artefactos.")
    parser.add_argument("--eval", action="store_true", help="Ejecutar métricas de evaluación de calidad tras el preprocesamiento (PSNR, SSIM).")
    
    args = parser.parse_args()

    # Determine if input is URL
    is_url = args.image_input.startswith("http://") or args.image_input.startswith("https://")
    
    temp_dir = None
    image_path = args.image_input

    try:
        if is_url:
            # Create a temporary file and download the image
            # Try to get the original filename from the URL, otherwise use a generic name
            filename = os.path.basename(urllib.parse.urlsplit(args.image_input).path)
            if not filename or not any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                filename = "downloaded_avatar.png"
                
            temp_dir = tempfile.TemporaryDirectory()
            image_path = os.path.join(temp_dir.name, filename)
            
            try:
                download_image(args.image_input, image_path)
            except Exception as e:
                logger.error(f"Error downloading image: {e}")
                sys.exit(1)
        
        artalk_path = os.path.abspath(args.artalk_path)
        
        logger.info("\n=== ARTalk Avatar Preprocessor ===")
        logger.info(f"Imagen:         {image_path}")
        logger.info(f"ARTalk repo:    {artalk_path}")
        logger.info(f"Dispositivo:    {args.device}")
        
        # Run the preprocessor
        logger.info("\nIniciando 'cocinado' del avatar...")
        avatar_id = preprocess_avatar_image(
            image_path_str=image_path,
            artalk_path_str=artalk_path,
            device=args.device,
            no_matting=args.no_matting
        )
        
        logger.info("\n=== ¡Proceso Completado Exitosamente! ===")
        logger.info(f"Avatar ID generado:        '{avatar_id}'")
        logger.info("\nInstrucciones de Uso:")
        logger.info("1. El tracker ha almacenado la configuración FLAME en el archivo:")
        logger.info(f"   {os.path.join(artalk_path, 'assets', 'GAGAvatar', 'tracked.pt')}")
        logger.info("2. Para usar este avatar en tu agente o inferencia, pásale el id generado como parámetro.")
        logger.info("   Por ejemplo, en Livekit Agents:")
        logger.info(f"   avatar = ARTalkAvatarSession(shape_id='{avatar_id}', ...)")
        logger.info("\n3. Si usas la App de Gradio original de ARTalk (inference.py), este ID ahora es reconocible si agregas la imagen a su lista de ejemplos, o lo especificas manualmente.")

        # Optional Evaluation
        if args.eval:
            import cv2
            import torch
            logger.info("\nIniciando evaluación de calidad...")
            
            # Load original image for comparison
            original_img = cv2.imread(image_path)
            
            # Load the result from tracked.pt
            tracked_pt_path = os.path.join(artalk_path, 'assets', 'GAGAvatar', 'tracked.pt')
            if os.path.exists(tracked_pt_path):
                tracked_db = torch.load(tracked_pt_path, map_location='cpu', weights_only=False)
                if avatar_id in tracked_db:
                    metrics = evaluate_avatar_quality(original_img, tracked_db[avatar_id])
                    display_metrics(metrics)
                else:
                    logger.error(f"Error: Avatar '{avatar_id}' no encontrado en la base de datos de tracking.")
            else:
                logger.error(f"Error: No se encontró el archivo de tracking en {tracked_pt_path}")

    except Exception as e:
        logger.error(f"\n[Error]: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup temp dir if it was created
        if temp_dir is not None:
            temp_dir.cleanup()

if __name__ == "__main__":
    main()
