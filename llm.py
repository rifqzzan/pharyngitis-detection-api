from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_explanation(prediction: str, confidence: float, language: str = "id") -> str:
    if language == "en":
        prompt = f"""
        You are a medical assistant helping explain diagnostic results.
        
        Throat image detection results show:
        - Prediction: {prediction}
        - Confidence level: {confidence}%
        
        Provide a brief explanation (3-4 sentences) in English that is easy for patients to understand:
        1. What this result means
        2. What should be done next
        
        Note: Remind that this is only a screening tool, not a substitute for a doctor's diagnosis.
        """
    else:
        prompt = f"""
        Kamu adalah asisten medis yang membantu menjelaskan hasil diagnosis.
        
        Hasil deteksi gambar tenggorokan menunjukkan:
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