from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_explanation(prediction: str, confidence: float) -> str:
    prompt = f"""
    Kamu adalah asisten medis yang membantu menjelaskan hasil diagnosis.
    
    Hasil deteksi gambar endoskopi menunjukkan:
    - Prediksi: {prediction}
    - Tingkat keyakinan: {confidence}%
    
    Berikan penjelasan singkat (3-4 kalimat) dalam Bahasa Indonesia yang mudah dipahami pasien tentang:
    1. Apa artinya hasil ini
    2. Apa yang sebaiknya dilakukan selanjutnya
    
    Catatan: Ingatkan bahwa ini hanya alat bantu screening, bukan pengganti diagnosis dokter.
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content