from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    print("Error: NVIDIA_API_KEY not found in environment.")
else:
    print(f"API Key found (length: {len(api_key)})")

try:
    print("Listing available models...")
    available = ChatNVIDIA.get_available_models()
    print("Available models:")
    for model in available:
        if "kimi" in model.id or "mistral" in model.id:
            print(f" - {model.id} ({model.model_type})")
            
    print("\nTesting Kimi connection...")
    llm = ChatNVIDIA(model="moonshotai/kimi-k2.5")
    result = llm.invoke("Hello, are you working?")
    print(f"Result: {result.content}")

except Exception as e:
    print(f"Error: {e}")
