#!/usr/bin/env python3
"""Medidor de nivel de audio en tiempo real. Muestra una barra mientras hablas.

Uso:
    python3.14 scripts/test_mic.py --level
"""

import sys
import numpy as np
import sounddevice as sd

BLOCK = "█"
BAR_WIDTH = 50


def main() -> None:
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append(i)
            print(f"  [{i:2d}] {dev['name']}")

    if not devices:
        print("No hay dispositivos de entrada.")
        return

    device = 5  # default
    if len(sys.argv) > 1:
        try:
            device = int(sys.argv[1])
        except ValueError:
            pass

    print(f"\nUsando dispositivo [{device}]. Habla para ver la barra. Ctrl+C para salir.\n")

    def callback(indata: np.ndarray, frames: int, time_info: dict, status: int) -> None:
        mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
        level = float(np.sqrt(np.mean((mono.astype(np.float32) / 32768.0) ** 2) + 1e-10))

        pct = min(level * 10, 1.0)  # scale for visibility
        bar_len = int(pct * BAR_WIDTH)
        bar = BLOCK * bar_len + " " * (BAR_WIDTH - bar_len)
        db = 20 * np.log10(level + 1e-10)
        status_text = ">>> HABLANDO <<<" if level > 0.04 else "   (silencio)"

        sys.stdout.write(f"\r[{bar}] {level:.4f} ({db:+.0f} dB) {status_text}  ")
        sys.stdout.flush()

    try:
        stream = sd.InputStream(device=device, samplerate=16000, channels=1, dtype="int16", callback=callback)
        stream.start()
        while stream.active:
            sd.sleep(100)
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()
        print()


if __name__ == "__main__":
    main()
