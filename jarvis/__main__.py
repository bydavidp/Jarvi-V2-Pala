import sys

from core.config import Config
from core.logs import setup_logging
from core.state import AssistantState


def main() -> None:
    logger = setup_logging()
    logger.info("Iniciando JARVIS v0.1.0")

    try:
        config = Config()
    except (FileNotFoundError, ValueError) as e:
        logger.error("Error al cargar configuración: %s", e)
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    estado = AssistantState.IDLE.value

    logger.info("Configuración cargada correctamente")
    print("JARVIS v0.1.0")
    print(f"  Idioma               : {config.idioma}")
    print(f"  Ollama URL            : {config.ollama.url}")
    print(f"  Modelo LLM            : {config.ollama.model}")
    print(f"  Whisper modelo        : {config.whisper.model}")
    print(f"  Piper modelo          : {config.piper.model_path}")
    print(f"  Piper binario         : {config.piper.binary_path}")
    print(f"  Wake word modelo      : {config.wakeword.model_path}")
    print(f"  Dispositivo audio in  : {config.audio.input_device or 'por defecto'}")
    print(f"  Dispositivo audio out : {config.audio.output_device or 'por defecto'}")
    print(f"  Apps en whitelist     : {', '.join(config.whitelist.apps.keys())}")
    print(f"  Estado                : {estado}")


if __name__ == "__main__":
    main()
