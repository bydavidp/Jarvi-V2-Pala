"""Diagnostico completo de fallos del matcher."""
import re, yaml
from pathlib import Path

patterns = yaml.safe_load(Path("brain/patterns.yaml").read_text(encoding="utf-8"))["patterns"]

phrases = [
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

cats = {}
for phrase, exp_skill, exp_op in phrases:
    matched = any(re.fullmatch(p["regex"], phrase) for p in patterns)
    if matched:
        continue

    causes = []
    pl = phrase.lower()

    vos_verbs = ["subele", "bajale", "apagame", "quitale", "abreme",
                 "buscame", "investigue", "ponle", "abri", "decime", "poneme"]
    found = [v for v in vos_verbs if v in pl]
    if found:
        causes.append("voseo:" + ",".join(found))

    for m in ["oye", "a ver", "hazme el favor", "necesito que"]:
        if pl.startswith(m):
            causes.append("muletilla-inicial:" + m)
            break

    first_word = pl.split()[0]
    all_verbs = ["sube", "baja", "pon", "abre", "busca", "silencia", "apaga",
                 "activa", "enciende", "entra", "decime", "poneme", "abri"]
    has_verb = any(pl.startswith(v) for v in all_verbs)
    if not has_verb and not found:
        causes.append("eliptica-sin-verbo")

    if "sonido" in pl:
        causes.append("sinonimo:sonido-no-volumen")
    if "ruido" in pl:
        causes.append("no-cubierto:ruido")
    if "poquito" in pl:
        causes.append("diminutivo:poquito")
    if "porfa" in pl:
        causes.append("abreviatura:porfa")
    if "todo" in pl and "volumen" in pl:
        causes.append("cuantificador:todo")
    if "mute" in pl:
        causes.append("anglicismo:mute")
    if "en 25" in pl:
        causes.append("preposicion:en-no-al")
    if "otra vez" in pl:
        causes.append("frase-compuesta")
    if "eso" in pl:
        causes.append("pronombre:eso")
    if "dia" in pl or ("fecha" in pl and not has_verb):
        causes.append("variante-dia-no-hora")
    if "entra a http" in pl:
        causes.append("verbo-no-cubierto:entra")

    print(f"  '{phrase[:45]:45s} -> {', '.join(causes) if causes else 'NO DIAGNOSTICADO'}")

    for c in causes:
        cat = c.split(":")[0]
        cats[cat] = cats.get(cat, 0) + 1

print(f"\n{'='*60}")
print("CONTEO POR CATEGORIA")
print(f"{'='*60}")
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {c:30s}: {n}")

# Simulacion sin anclaje
print(f"\n{'='*60}")
print("SIMULACION: sin anclaje (re.search, no ^ ni $)")
print(f"{'='*60}")
ok = 0
bad = 0
miss = 0
for phrase, exp_skill, exp_op in phrases:
    found_correct = False
    found_wrong = False
    for p in patterns:
        regex_loose = p["regex"].replace(r"^\s*", "").replace(r"\s*$", "")
        if re.search(regex_loose, phrase):
            if p["skill"] == exp_skill and p["operation"] == exp_op:
                found_correct = True
            else:
                found_wrong = True
            break
    if found_correct:
        ok += 1
    elif found_wrong:
        bad += 1
        print(f"  BAD  \"{phrase}\"")
    else:
        miss += 1
        print(f"  MISS \"{phrase}\"")

pct = round(ok / 20 * 100)
print(f"\n  Con anclaje:    1/20 (5 pct)")
print(f"  Sin anclaje:    {ok}/20 ({pct} pct) correctos, {bad} incorrectos, {miss} sin match")
