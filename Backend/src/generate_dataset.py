import random
import csv

positives = [
    "excelente", "muy bueno", "me encanta", "maravilloso", "perfecto",
    "recomendado", "feliz con la compra", "fantástico", "lo volvería a comprar",
    "satisfecho", "increíble", "estupendo", "de primera calidad",
    "bueno y bonito", "exelente", "super contento", "vale la pena"
]

negatives = [
    "horrible", "muy malo", "decepcionante", "asqueroso", "lo odio",
    "pésimo", "nunca más", "terrible", "desastre", "fatal", "el peor",
    "no me gustó", "bno", "orrible", "frustrante", "desilusionante"
]

positive_sentences = []
negative_sentences = []

for _ in range(500):
    prefix = random.choice([
        "Este producto es", "Servicio", "Compra", "Entrega", "La experiencia fue", 
        "Atención al cliente", "Calidad", "Lo que recibí fue"
    ])
    suffix = random.choice(positives)
    positive_sentences.append([f"{prefix} {suffix}", 1])

for _ in range(500):
    prefix = random.choice([
        "Este producto es", "Servicio", "Compra", "Entrega", "La experiencia fue", 
        "Atención al cliente", "Calidad", "Lo que recibí fue"
    ])
    suffix = random.choice(negatives)
    negative_sentences.append([f"{prefix} {suffix}", 0])

# Mezclar dataset
dataset = positive_sentences + negative_sentences
random.shuffle(dataset)

# Guardar en CSV
with open('../data/reviews.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['text', 'label'])
    writer.writerows(dataset)

print(f"✅ Dataset generado con {len(dataset)} registros en ../data/reviews.csv")
