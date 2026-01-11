import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Asegúrate de tener tu API KEY configurada
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

    print("Listado de modelos disponibles para ti:")
    print("-" * 40)

    for m in genai.list_models():
        # Filtramos para ver solo los que generan contenido (chat) y embeddings
        if 'generateContent' in m.supported_generation_methods:
            print(f"Nombre: {m.name}")
            print(f"   -> ID para usar: {m.name.replace('models/', '')}")
            print(f"   -> Descripción: {m.description}")
            print("-" * 40)


if __name__ == "__main__":
    main()


