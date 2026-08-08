#!/usr/bin/env python3
"""Medidor de nivel calibrado: mide silencio y habla, recomienda umbral.

Uso:
    python3.14 scripts/level_meter.py [dispositivo]
"""

import sys
import time
import numpy as np
import sounddevice as sd

BLOCK = chr(9608)
BAR_WIDTH = 50


def measure_phase(label: str, seconds: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  FASE {label}: {seconds}s — {'NO hables' if 'SILENCIO' in label else 'HABLA normalmente'}")
    print(f"{'='*60}")

    rms_values: list[float] = []

    def callback(indata, frames, time_info, status):
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
        level = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2) + 1e-10))
        rms_values.append(level)

        pct = min(level * 10, 1.0)
        bar_len = int(pct * BAR_WIDTH)
        bar = BLOCK * bar_len + " " * (BAR_WIDTH - bar_len)
        text = ">>> HABLANDO <<<" if level > 0.05 else "   (silencio)    "
        sys.stdout.write(f"\r[{bar}] {level:.4f}  {text}  ")
        sys.stdout.flush()

    stream = sd.InputStream(device=args_device(), samplerate=16000, channels=1, dtype="int16", callback=callback)
    stream.start()
    sd.sleep(seconds * 1000)
    stream.stop()
    stream.close()
    print()

    arr = np.array(rms_values)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def args_device() -> int | None:
    if len(sys.argv) > 1:
        try:
            return int(sys.argv[1])
        except ValueError:
            pass
    return None


def main():
    device = args_device()
    if device is not None:
        print(f"Dispositivo: [{device}]\n")

    silence = measure_phase("SILENCIO", 5)
    speech = measure_phase("HABLA", 5)

    print(f"\n{'='*60}")
    print(f"{'ESTADISTICA':<12} {'SILENCIO':>10} {'HABLA':>10}")
    print(f"{'='*60}")
    for key in ["min", "mean", "p50", "p95", "p99", "max"]:
        print(f"  {key:<10} {silence[key]:10.5f} {speech[key]:10.5f}")

    # Criterio: 30% del camino entre p95(silencio) y min(habla), sesgado a silencio
    p5_speech = speech["p50"] * 0.5 if speech["min"] < silence["p95"] else speech["min"]
    if speech["min"] > silence["p95"]:
        gap = speech["min"] - silence["p95"]
        recommended = silence["p95"] + gap * 0.3
    else:
        recommended = silence["p95"] * 1.2

    print(f"\n  Umbral recomendado: {recommended:.5f}")
    print(f"  Criterio: 30% entre p95(silencio)={silence['p95']:.5f} y min(habla)={speech['min']:.5f}")


if __name__ == "__main__":
    main()
