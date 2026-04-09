
import os
import asyncio
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found in environment.")
    exit(1)

async def check_models():
    print(f"Checking Gemini models with key ending in ...{api_key[-5:]}")
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})
        
        print("\n--- Listing Available Models (v1beta) ---")
        try:
            # List models using the correct method for the new SDK
            # It returns a pager/iterator
            pager = await client.aio.models.list()
            async for model in pager:
                print(f"Name: {model.name}")
                print(f"  DisplayName: {model.display_name}")
                print(f"  SupportedMethods: {model.supported_generation_methods}")
                print("-" * 20)
                
        except Exception as e:
            print(f"Failed to list models: {e}")

    except ImportError:
        print("google-genai library not found or import error.")
    except Exception as e:
        print(f"General error during check: {e}")

if __name__ == "__main__":
    asyncio.run(check_models())
