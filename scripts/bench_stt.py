"""Benchmark de STT con validacion de audio sano y opcion de guardar WAV."""
import argparse
import time
import wave
import numpy as np
from audio.stt import SttEngine
from audio.capture import AudioCapture
from core.config import WhisperConfig

MIN_SPEECH_RATIO = 0.4

parser = argparse.ArgumentParser()
parser.add_argument("--save-wav", action="store_true", help="Guardar audio capturado a tmp/captura.wav")
args = parser.parse_args()

print("Grabando frase de prueba...")
cap = AudioCapture(device=5, sample_rate=16000, vad_silence_ms=800, vad_min_speech_ms=200, vad_threshold=0.5)
result = cap.capture_phrase(timeout_seconds=10)

if not result.get("audio"):
    print("ERROR: No se detecto habla.")
    exit(1)

speech_ms = result.get("speech_ms", 0)
total_ms = result["duration"] * 1000
ratio = speech_ms / total_ms if total_ms > 0 else 0

if ratio < MIN_SPEECH_RATIO:
    print(f"ERROR: Solo {speech_ms:.0f}ms de habla en {total_ms:.0f}ms ({ratio:.0%}) < {MIN_SPEECH_RATIO:.0%}")
    exit(1)

audio = result["audio"]
print(f"Audio OK: {result['duration']:.2f}s, habla detectada: {speech_ms:.0f}ms ({ratio:.0%})")

# Reportar nivel de audio ANTES de normalizar
raw = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
peak = float(np.abs(raw).max())
rms = float(np.sqrt(np.mean(raw**2) + 1e-10))
print(f"Nivel PRE:  peak={peak:.4f}, RMS={rms:.4f} (escala 0-1)")
if peak < 0.2:
    print("  ADVERTENCIA: audio bajo, se normalizara. Revisa ganancia del micro en Windows.\n")
else:
    print()

if args.save_wav:
    import numpy as np
    from pathlib import Path
    Path("tmp").mkdir(exist_ok=True)
    wav_path = "tmp/captura.wav"
    raw = np.frombuffer(audio, dtype=np.int16)
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(raw.tobytes())
    print(f"Audio guardado: {wav_path} ({len(raw)} muestras, pico={raw.max()})\n")

for model_name in ["base", "small"]:
    for beam in [1]:
        cfg = WhisperConfig(model=model_name, compute_type="int8", language="es")
        engine = SttEngine(cfg)
        engine.transcribe(audio)  # warmup + carga

        times = []
        for i in range(3):
            t0 = time.perf_counter()
            stt = engine.transcribe(audio)
            ms = (time.perf_counter() - t0) * 1000
            times.append(ms)

        avg = sum(times[1:]) / 2
        text = stt["text"]
        print(f"  {model_name:6s} beam={beam}: {avg:6.0f}ms | \"{text[:80]}\"")

if not args.save_wav:
    print("\nPara guardar el audio y verificarlo: python3.14 scripts/bench_stt.py --save-wav")
