# AGENTS.md — Proyecto JARVIS (asistente local por voz)

> Este archivo es contexto permanente para el agente. Léelo completo antes de cualquier cambio.
> Si una instrucción del usuario contradice este archivo, **pregunta antes de actuar**.

---

## 1. Qué estamos construyendo

Un asistente personal **100% local** que:

- Se activa por palabra clave ("Jarvis").
- Escucha, transcribe, entiende y responde **por voz**.
- Tiene una **cara animada** en pantalla que reacciona al estado (idle / escuchando / pensando / hablando).
- Ejecuta acciones en el PC: abrir programas, abrir navegador y buscar, subir/bajar volumen, decir la hora, controlar ventanas.
- Puede investigar en internet y dar opiniones usando un LLM.
- Puede conectarse a dispositivos de la red local (fase avanzada).
- **Nunca** toca configuraciones sensibles del sistema sin autorización explícita.

Objetivo de diseño: **modular**. Cada capa debe poder probarse y reemplazarse sin tocar las demás.

### Fases intermedias planificadas

- **Fase 6.5** — Clasificador semántico como capa 2 entre matcher y LLM:
  `sentence-transformers` con `paraphrase-multilingual-MiniLM-L12-v2` (~120MB,
  CPU, <100ms). Ejemplos por intención en YAML. Umbral de confianza configurable.
  El clasificador devuelve SOLO la intención, nunca parámetros. Latencia objetivo
  <100ms. Solo se ejecuta si el matcher no resolvió.

---

## 2. Stack fijo (no cambiar sin preguntar)

| Capa | Tecnología |
|---|---|
| Backend | Python 3.11+ · FastAPI · Uvicorn |
| STT | `faster-whisper` (modelo `small` o `base`, español) |
| Wake word | `openWakeWord` |
| VAD | `webrtcvad` o `silero-vad` |
| LLM | Ollama (HTTP local `http://localhost:11434`) |
| TTS | Piper (voz española) |
| Control del SO | `subprocess`, `psutil`, `pycaw` (Windows) / `pactl` (Linux), `pyautogui` |
| UI / avatar | React + Vite + TypeScript, conectada por WebSocket |
| Persistencia | SQLite vía SQLAlchemy |
| Config | YAML en `config/` + `.env` para secretos |
| Tests | pytest |

Reglas de dependencias:
- No agregues una librería nueva sin justificarla en el mensaje de salida.
- Nada de servicios en la nube salvo búsqueda web explícita.
- Todo debe correr sin internet excepto las skills de búsqueda.

---

## 3. Estructura de carpetas (respétala)

```
jarvis/
├── AGENTS.md
├── pyproject.toml
├── .env.example
├── config/
│   ├── settings.yaml          # modelos, rutas, idioma, dispositivos de audio
│   └── permissions.yaml       # política de permisos por skill
├── core/
│   ├── __init__.py
│   ├── bus.py                 # event bus interno (pub/sub asíncrono)
│   ├── orchestrator.py        # máquina de estados del asistente
│   ├── state.py               # enum de estados + estado global
│   └── logs.py                # logging estructurado
├── jarvis/
│   └── __main__.py            # entry point: python -m jarvis
├── audio/
│   ├── __init__.py
│   ├── capture.py             # micrófono + VAD
│   ├── wakeword.py
│   ├── stt.py
│   └── tts.py
├── brain/
│   ├── __init__.py
│   ├── llm.py                 # cliente Ollama
│   ├── router.py              # clasificación de intención
│   ├── prompts.py             # system prompts del asistente
│   └── memory.py              # memoria corta + larga
├── skills/
│   ├── __init__.py
│   ├── base.py                # clase Skill + registry + decorador @skill
│   ├── time_skill.py
│   ├── volume.py
│   ├── apps.py
│   ├── browser.py
│   ├── websearch.py
│   └── devices.py
├── security/
│   ├── __init__.py
│   ├── policy.py              # evaluación de permisos
│   ├── confirm.py             # flujo de confirmación por voz / PIN
│   └── audit.py               # log inmutable de acciones ejecutadas
├── api/
│   ├── __init__.py
│   └── server.py              # FastAPI + WebSocket para la UI
├── ui/                        # React (avatar)
├── tests/
│   └── __init__.py
└── scripts/
```

---

## 4. Política de permisos (el corazón del proyecto)

Cada **operación** declara su nivel base, no la clase de skill completa.
`security/policy.py` es el **único** punto que autoriza ejecución, y **evalúa el
nivel por invocación**: los parámetros concretos de cada llamada pueden elevar el
nivel efectivo, pero nunca bajarlo. Ejemplo: `volume.get()` declara base SAFE,
`volume.set()` y `volume.mute()` declaran base SYSTEM. Si `volume.set(100)` fuera
extremo, el evaluador podría subirlo a SENSITIVE, pero `volume.get()` jamás baja
a SAFE desde SYSTEM porque su base ya es SAFE.

| Nivel | Qué incluye | Comportamiento |
|---|---|---|
| `SAFE` | hora, clima, conversación, búsqueda web, abrir navegador, reproducir música | Ejecuta directo |
| `SYSTEM` | volumen, brillo, abrir/cerrar apps de usuario, control de ventanas, screenshots | Ejecuta + registra en audit log |
| `SENSITIVE` | archivos del usuario, red, dispositivos externos, instalación de software, procesos del sistema | Requiere confirmación explícita (voz + PIN) |
| `FORBIDDEN` | `sudo`/admin, registro de Windows, firewall, credenciales, formateo, borrado recursivo, desactivar antivirus, modificar la propia capa de permisos | **Se rechaza siempre**, sin excepción |

Reglas duras para el agente:

1. **Nunca** escribas código que ejecute comandos con `sudo`, `runas`, o privilegios elevados.
2. **Nunca** hagas que una skill salte `policy.check()`. Toda ejecución pasa por ahí.
3. **Nunca** construyas comandos de shell concatenando texto del LLM o del usuario. Usa listas de argumentos y whitelists.
4. Las apps que se pueden abrir vienen de una **whitelist** en `config/settings.yaml`, no de texto libre.
5. El PIN va en `.env`, nunca hardcodeado, nunca en logs.
6. Si una tarea te obliga a debilitar esta política, **para y pregunta**.

---

## 5. Convenciones de código

- Type hints en todo. `mypy` limpio en `core/`, `security/`, `skills/`.
- Async por defecto en I/O (audio, HTTP, WebSocket).
- Nada de estado global mutable fuera de `core/state.py`.
- Cada skill: una clase que hereda de `Skill`, con `name`, `level`, `patterns`/`description` y `async def run(...)`.
- Errores: nunca `except: pass`. Loguea y devuelve un resultado degradado.
- Comentarios y docstrings en español; nombres de variables y funciones en inglés.
- Config por YAML, secretos por `.env`. Cero rutas absolutas hardcodeadas.

---

## 6. Comandos del proyecto

```bash
uv sync                      # o: pip install -e .
pytest -q                    # tests
python -m jarvis             # arrancar el asistente
uvicorn api.server:app --reload   # solo backend
cd ui && npm run dev         # solo UI
```

---

## 7. Cómo debes trabajar (reglas del loop)

1. **Alcance cerrado**: toca solo los archivos que la tarea nombra. Si necesitas tocar otro, dilo antes.
2. **Un paso a la vez**: no adelantes fases futuras "porque ya que estamos".
3. **Verifica antes de reportar**: corre los tests o el script de la fase. Si no puedes ejecutarlo, dilo explícitamente.
4. **Reporta en este formato** al terminar:
   - Archivos creados/modificados
   - Qué hace ahora el sistema que antes no hacía
   - Cómo lo probé y qué salió
   - Qué quedó pendiente o dudoso
5. **No hagas commit hasta que yo diga OK.** Cuando lo diga, un commit por fase con mensaje convencional (`feat(audio): ...`).
6. Si el requerimiento es ambiguo, **haz máximo 3 preguntas** y espera. No inventes.

---

## 8. Decisiones arquitectónicas (cerradas)

Estas decisiones están tomadas y no se reabren sin orden explícita.

### 8.1 Event bus: in-process asyncio

El bus es `asyncio.Queue` en proceso. El wake word corre en un **hilo**
(`threading.Thread`), no en un proceso aparte.

- **Justificación**: `onnxruntime` libera el GIL durante la inferencia, por lo que un hilo
  basta para no bloquear el loop asíncrono.
- **Interfaz pública**:
  - `bus.publish(event)`: solo para publicadores que corren en el event loop (corrutinas).
  - `bus.publish_threadsafe(event)`: obligatorio para publicadores desde hilos. Usa
    `loop.call_soon_threadsafe()` internamente porque `asyncio.Queue` no es thread-safe.
  - `bus.subscribe(pattern, callback)`: registra un callback asíncrono.
  - Los llamadores nunca acceden a la cola directamente. Esto permite migrar a
    Redis/RabbitMQ en el futuro sin tocar publicadores ni suscriptores.
- **Qué productor usa cada método**:
  | Productor | Contexto | Método |
  |---|---|---|
  | `audio/wakeword.py` | Hilo (`threading.Thread`) | `publish_threadsafe()` |
  | `audio/capture.py` | Hilo (sounddevice callback) | `publish_threadsafe()` |
  | `core/orchestrator.py` | Event loop | `publish()` |
  | `audio/tts.py` | Event loop | `publish()` |
  | `brain/llm.py` | Event loop | `publish()` |
  | `skills/*.py` | Event loop (despachadas por orchestrator) | `publish()` |
- **Qué NO se usa**: Redis, RabbitMQ, multiprocessing, ni ningún broker externo.

### 8.2 Routing: matcher determinista + LLM como fallback

Dos etapas, en este orden:

1. **Matcher determinista**: patrones y palabras clave predefinidos (regex, keywords)
   para comandos frecuentes y predecibles ("qué hora es", "sube el volumen").
   Si hay match, se despacha la skill directamente.
2. **LLM**: solo si el matcher no resolvió nada, se consulta a Ollama. La respuesta
   del LLM debe ser **JSON estricto** validado con pydantic. Si el JSON es inválido,
   se reintenta una vez con un prompt de corrección; si vuelve a fallar, se devuelve
   un error hablado genérico.

- **Qué NO se usa**: function-calling de Ollama, respuestas en texto libre del LLM
  para routing.

### 8.3 Permisos: evaluación por invocación

`policy.check(skill_name, params)` evalúa el nivel **para cada llamada concreta**,
no para la clase de skill. La misma skill puede resolverse a niveles distintos según
sus parámetros:

- `volume.get()` → SAFE
- `volume.set(50)` → SYSTEM
- `volume.set(100)` → podría escalarse a SENSITIVE si el valor es extremo

El nivel base declarado en la skill es el piso; el evaluador solo puede **subir**
el nivel, nunca bajarlo.

### 8.4 Audio duplex: half-duplex con margen

Mientras el estado es `SPEAKING`, el wake word y la captura de micrófono quedan
**suspendidos**, con un margen de 200ms tras `SPEAKING_END` para evitar que el eco
del altavoz reactive al asistente.

- La interrupción del TTS en ejecución se hace **por tecla** (configurable en
  `settings.yaml`), no por voz, en la v1 del sistema.
- La interrupción por voz queda como mejora futura.

### 8.5 Piper en Windows: binario, no paquete PyPI

Piper se ejecuta invocando el binario descargado del
[GitHub release](https://github.com/rhasspy/piper/releases), **no** el paquete de PyPI
(`piper-tts`).

- El binario se descarga con `scripts/download_models.py` y se guarda en una ruta
  configurable en `settings.yaml`.
- Se invoca con `subprocess.run([ruta_binario, ...], capture_output=True)`, argumentos
  como lista, sin `shell=True`.
- **Justificación**: el paquete de PyPI tiene dependencia de `piper-phonemize` que
  compila desde C++ y falla frecuentemente en Windows. El binario precompilado del
  release es más fiable.

---

## 9. Fronteras de tipos

Reglas que gobiernan qué tan estricto es el tipado en cada capa y qué contratos
cruzan los límites entre módulos.

### 9.1 Niveles de exigencia por paquete

| Paquete | Nivel | Regla |
|---|---|---|
| `core/` | **Estricto** | Type hints en todas las funciones públicas. `mypy --strict` debe pasar antes de merge. |
| `security/` | **Estricto** | Type hints en todas las funciones públicas. `mypy --strict` debe pasar. Sin `Any` en firmas públicas. |
| `skills/` | **Estricto** | Type hints en `run()` y en los parámetros de cada skill. `SkillResult` como tipo de retorno obligatorio. |
| `brain/` | **Moderado** | Type hints en funciones públicas. Se tolera `Any` en payloads de Ollama (el schema depende del modelo). |
| `audio/` | **Moderado** | Type hints en funciones públicas. Callbacks de sounddevice pueden usar `Any` para buffers numpy. |
| `api/` | **Moderado** | Type hints en endpoints y modelos de respuesta. |
| `ui/` | **TypeScript estricto** | `tsc --noEmit` debe pasar. Sin `any` salvo en payloads de WebSocket. |

### 9.2 Contratos entre capas (qué cruza cada frontera)

| Frontera | Tipo que cruza | Dirección |
|---|---|---|
| `audio/` → `core/` | `Event` (tipo `stt_result`, `wakeword_detected`) | Audio publica eventos al bus |
| `core/` → `skills/` | `SkillRequest` (nombre + dict params) | Orchestrator despacha a skills |
| `skills/` → `core/` | `SkillResult` (success, data, error) | Skill devuelve resultado |
| `brain/` → `core/` | `RouterDecision` (action: CHAT/SKILL/SEARCH, payload) | Router clasifica intención |
| `core/` → `api/` | `Event` (todos los tipos) | Bus → WebSocket broadcast |
| `core/` → `audio/` | `str` (texto a hablar) | Orchestrator → TTS |

Ninguna capa importa directamente de una capa superior. Las dependencias van:
`skills/` → `security/` → `core/` ← `brain/` ← `audio/`

---

## 10. Pendientes conocidos

### 10.1 Deuda técnica agendada

Bugs latentes o riesgos técnicos con fase asignada para resolverlos.

| Item | Archivo esperado | Fase | Arreglo |
|---|---|---|---|
| `compute_type="auto"` en Windows sin GPU NVIDIA | `audio/stt.py` | Fase 5 | Si no se detecta GPU NVIDIA (`torch.cuda.is_available()` o similar), forzar `compute_type="int8"`. `"auto"` en CPU elige `float32` y dispara latencia 3-5x. |
| COM no inicializado en hilo de skills | `skills/volume.py` | Fase 9 | `pycaw` depende de `comtypes.CoInitialize()`. Si el hilo que ejecuta `volume.set()` no llamó a `CoInitialize`, falla con `COMError`. La skill debe llamar `pythoncom.CoInitialize()` al entrar y `CoUninitialize()` al salir. |
| Modelo STT: base vs small | `audio/stt.py` + `config/settings.yaml` | Fase 5 | Se eligió `base` con beam_size=1, condition_on_previous_text=False. Latencia ~1.7s en esta máquina. Small da ~5.4s (3x) sin mejora de precisión suficiente para justificarlo. Si la precisión de base resulta insuficiente en uso real, reevaluar small. |
| Router LLM: 31s de silencio en fallback | `core/orchestrator.py` | Fase 7 | Cuando el router degrade a LLM, emitir de inmediato una respuesta hablada de espera ("dame un momento") antes de consultar. El timeout del LLM debe producir SIEMPRE una respuesta hablada de error, nunca silencio. |
| Despachador no fuerza `decision.resolved_params` | `core/orchestrator.py` + `skills/base.py` | Fase 3 | **Defensa en profundidad, no imposibilidad técnica.** |** La estrategia es: (1) `Skill._execute()` valida un token de autorización por invocación emitido por el registry; (2) si falta o es inválido, lanza `SkillAuthError` y registra intento de bypass en audit log; (3) el registry construye los argumentos exclusivamente desde `decision.resolved_params`. La garantía real es code review + audit log; el token detecta el bypass, no lo previene. |

### 10.2 Fuera del MVP

Lo que **no** se va a implementar en la v1 y queda registrado para no discutirlo
en cada fase:

| Pendiente | Motivo | Fase prevista |
|---|---|---|
| Interrupción por voz durante TTS | Complejidad de echo cancellation en half-duplex. La v1 usa tecla. | Post v1 |
| Soporte Linux nativo (no solo detección runtime) | El SO objetivo es Windows 11. Adaptadores de SO se escriben con `if platform == "linux"` pero no se prueban. | Post v1 |
| Múltiples wake words ("Jarvis", "Asistente", etc.) | openWakeWord puede cargar varios modelos pero añade latencia y falsos positivos. | v1.1 |
| Idiomas además de español | Whisper y Piper lo soportan, pero los prompts y validaciones están hardcodeados en español. | v1.1 |
| Streaming de audio a la UI | La UI recibe solo el texto final transcrito, no audio. | Post v1 |
| Hot-reload de skills | Las skills se cargan al inicio. Añadir/editar skills requiere reinicio. | v1.2 |
| Cifrado del audit log | El log es SQLite en texto plano. | v1.1 |
| Modo offline absoluto (sin internet para búsqueda) | La búsqueda web requiere internet por definición. | N/A |
| Docker / contenedores | El acceso a micrófono, altavoz y GPU desde contenedores en Windows es inviable. | No planeado |
