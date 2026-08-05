"""Evaluacion honesta con frases colombianas y negativas."""
import asyncio
from unittest import mock

from brain.llm import LlmResponse
from brain.router import Router

POSITIVES = [
    ("Oye, que horas son", "time", "get_current_time"),
    ("Hazme el favor y dime la hora", "time", "get_current_time"),
    ("Subele un poquito al volumen", "volume", "adjust"),
    ("El volumen a 50", "volume", "set"),
    ("Bajale al sonido", "volume", "adjust"),
    ("Apagame ese ruido", "volume", "mute"),
    ("Quitale el mute", "volume", "unmute"),
    ("Abreme el Spotify, porfa", "apps", "open"),
    ("Necesito que me abras chrome", "apps", "open"),
    ("Buscame en Google como hacer empanadas", "browser", "search"),
    ("Investigue que es eso del machine learning", "browser", "search"),
    ("A ver, que dia es hoy?", "time", "get_current_time"),
    ("Subele todo el volumen", "volume", "adjust"),
    ("Silencia eso", "volume", "mute"),
    ("Ponle sonido otra vez", "volume", "unmute"),
    ("Abri el bloc de notas", "apps", "open"),
    ("Busca vuelos baratos a Cartagena", "browser", "search"),
    ("Decime la fecha de hoy", "time", "get_current_time"),
    ("Poneme el volumen en 25", "volume", "set"),
    ("Entra a https://youtube.com", "browser", "open_url"),
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


async def main():
    # Create client that returns CHAT for any non-matcher phrase
    class FakeClient:
        async def generate(self, **kwargs):
            return LlmResponse(text='{"intent": "CHAT"}')

    router = Router(FakeClient())

    print("=" * 60)
    print("FRASES POSITIVAS (esperado: matcher)")
    print("=" * 60)
    ok = 0
    for phrase, exp_skill, exp_op in POSITIVES:
        d = await router.route(phrase)
        hit = d.resolved_by == "matcher" and d.skill_name == exp_skill and d.operation == exp_op
        status = "OK " if hit else "FAIL"
        if not hit:
            print(f"  {status} '{phrase}' -> {d.resolved_by}/{d.skill_name}.{d.operation} (esperado {exp_skill}.{exp_op})")
        else:
            print(f"  {status} '{phrase}'")
        ok += 1 if hit else 0
    pct = round(ok / len(POSITIVES) * 100)
    print(f"\n  POSITIVOS: {ok}/{len(POSITIVES)} = {pct}%")

    print(f"\n{'='*60}")
    print("FRASES NEGATIVAS (esperado: NO matcher, NO skill)")
    print("=" * 60)
    fp = 0
    for phrase in NEGATIVES:
        d = await router.route(phrase)
        if d.resolved_by == "matcher" and d.skill_name:
            fp += 1
            print(f"  FALSE+ '{phrase}' -> matcher/{d.skill_name}.{d.operation}")
        else:
            print(f"  OK   '{phrase}' -> {d.resolved_by}")

    print(f"\n  FALSOS POSITIVOS: {fp}/{len(NEGATIVES)}")
    print(f"\n  RESULTADO FINAL: {pct}% positivos, {fp} falsos positivos")


asyncio.run(main())
