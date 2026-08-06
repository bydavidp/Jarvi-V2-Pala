"""Evaluacion del clasificador semantico (capa 2)."""
import time
from brain.classifier import IntentClassifier
from brain.router import IntentType

POSITIVES = [
    ("Oye, que horas son", "time"),
    ("Hazme el favor y dime la hora", "time"),
    ("Subele un poquito al volumen", "volume_adjust"),
    ("El volumen a 50", "volume_set"),
    ("Bajale al sonido", "volume_adjust"),
    ("Apagame ese ruido", "mute"),
    ("Quitale el mute", "unmute"),
    ("Abreme el Spotify, porfa", "apps_open"),
    ("Necesito que me abras chrome", "apps_open"),
    ("Buscame en Google como hacer empanadas", "browser_search"),
    ("Investigue que es eso del machine learning", "browser_search"),
    ("A ver, que dia es hoy?", "date"),
    ("Subele todo el volumen", "volume_adjust"),
    ("Silencia eso", "mute"),
    ("Ponle sonido otra vez", "unmute"),
    ("Abri el bloc de notas", "apps_open"),
    ("Busca vuelos baratos a Cartagena", "browser_search"),
    ("Decime la fecha de hoy", "date"),
    ("Poneme el volumen en 25", "volume_set"),
    ("Entra a https://youtube.com", "browser_open_url"),
]

NEGATIVES = [
    "apaga el computador",
    "borra ese archivo",
    "cierra sesion",
    "formatea el disco",
    "elimina todos los documentos",
    "desinstala el antivirus",
    "cambia la contrasena",
    "abre el administrador de tareas",
    "ejecuta virus.exe",
    "descarga torrents",
]

# Cargar clasificador
t0 = time.perf_counter()
clf = IntentClassifier(threshold=0.60)
load_ms = (time.perf_counter() - t0) * 1000
print(f"Clasificador cargado en {load_ms:.0f}ms\n")

# Evaluar positivos
print("=" * 60)
print("POSITIVOS")
print("=" * 60)
ok = 0
times = []
for phrase, expected in POSITIVES:
    t0 = time.perf_counter()
    r = clf.classify(phrase)
    lat = (time.perf_counter() - t0) * 1000
    times.append(lat)
    if r and r["intent"] == expected:
        ok += 1
        print(f"  OK  ({lat:.0f}ms) '{phrase}'")
    elif r:
        print(f"  WRONG ({lat:.0f}ms) '{phrase}' -> {r['intent']} (esperado {expected})")
    else:
        print(f"  MISS ({lat:.0f}ms) '{phrase}' -> sin clasificar")
pct = round(ok / len(POSITIVES) * 100)
avg_lat = sum(times) / len(times) if times else 0
print(f"\n  POSITIVOS: {ok}/{len(POSITIVES)} = {pct}% | latencia avg: {avg_lat:.0f}ms")

# Evaluar negativos
print(f"\n{'='*60}")
print("NEGATIVOS (debe ser rechazar o sin clasificar)")
print("=" * 60)
fp = 0
for phrase in NEGATIVES:
    r = clf.classify(phrase)
    if r and r["intent"]:
        if r["intent"] == "rechazar":
            print(f"  OK   '{phrase}' -> rechazar ({r['score']:.2f})")
        else:
            fp += 1
            print(f"  FALSE+ '{phrase}' -> {r['intent']} ({r['score']:.2f})")
    else:
        print(f"  OK   '{phrase}' -> sin clasificar")
print(f"\n  FALSOS POSITIVOS: {fp}/{len(NEGATIVES)}")

print(f"\n  RESULTADO: {pct}% positivos, {fp} FP, {avg_lat:.0f}ms latencia media")
