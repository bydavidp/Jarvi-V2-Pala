#!/usr/bin/env python3
"""Prueba de micrófono: lista dispositivos, graba, transcribe y reporta latencia.

Uso:
    python scripts/test_mic.py
    python scripts/test_mic.py --model small
    python scripts/test_mic.py --device 2
"""

import argparse
import sys
import time

from audio.capture import AudioCapture
from audio.stt import SttEngine
from core.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Test de micrófono y STT")
    parser.add_argument("--model", default="base", help="Modelo whisper (base, small, medium)")
    parser.add_argument("--device", type=int, default=None, help="Índice del dispositivo de audio")
    args = parser.parse_args()

    config = Config()

    # ── Listar dispositivos ──────────────────────────────────────────
    print("=" * 60)
    print("DISPOSITIVOS DE AUDIO DISPONIBLES")
    print("=" * 60)
    devices = AudioCapture.list_devices()
    for dev in devices:
        io = ""
        if dev["max_input_channels"] > 0:
            io += "IN "
        if dev["max_output_channels"] > 0:
            io += "OUT"
        if not io:
            io = "--"
        print(f"  [{dev['index']:2d}] {io:4s} {dev['name']} ({dev['default_samplerate']} Hz)")

    # ── Seleccionar dispositivo ──────────────────────────────────────
    device = args.device
    if device is None:
        default = sd.default.device[0] if hasattr(sd, "default") else None
        print(f"\nDispositivo de entrada por defecto: [{default}]")

    # ── Configurar captura ───────────────────────────────────────────
    capture = AudioCapture(
        device=device,
        sample_rate=config.audio.sample_rate,
        vad_silence_ms=config.audio.vad_silence_ms,
        vad_min_speech_ms=config.audio.vad_min_speech_ms,
        vad_threshold=config.audio.vad_threshold,
    )

    # ── Grabar ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("HABLA AHORA (esperando voz... habla y haz silencio para terminar)")
    print("=" * 60)

    total_start = time.perf_counter()
    result = capture.capture_phrase(timeout_seconds=10)

    if not result["audio"]:
        print("No se detectó habla o la frase fue demasiado corta.")
        sys.exit(1)

    print(f"\n[Captura] {result['duration']:.2f}s de audio ({len(result['audio'])} bytes)")
    print(f"[Captura] Latencia: {result['capture_latency']*1000:.0f}ms")
    print(f"[VAD]     Speech frames: {result.get('speech_frames', 0)}")

    # ── Transcribir ──────────────────────────────────────────────────
    whisper_config = config.whisper
    whisper_config.model = args.model

    print(f"\n[STT] Modelo: {whisper_config.model}, compute_type: {whisper_config.compute_type}")
    stt = SttEngine(whisper_config)

    transcribe_start = time.perf_counter()
    stt_result = stt.transcribe(result["audio"], sample_rate=config.audio.sample_rate)
    transcribe_end = time.perf_counter()

    # ── Resultados ───────────────────────────────────────────────────
    total_latency = time.perf_counter() - total_start

    print(f"\n{'='*60}")
    print("RESULTADO")
    print(f"{'='*60}")
    print(f"  Texto:       \"{stt_result['text']}\"")
    print(f"  Confianza:   {stt_result['confidence']:.3f}")
    print(f"  Idioma:      {stt_result['language']}")
    print(f"")
    print(f"  DESGLOSE DE LATENCIA:")
    print(f"    Captura:         {result['capture_latency']*1000:7.0f}ms")
    print(f"    Transcripción:   {stt_result['transcription_latency']*1000:7.0f}ms")
    print(f"    Total (pared):   {total_latency*1000:7.0f}ms")


if __name__ == "__main__":
    import sounddevice as sd
    main()
