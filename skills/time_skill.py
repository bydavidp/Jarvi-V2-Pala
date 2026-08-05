from datetime import datetime

from skills.base import Skill, SkillResult


class TimeSkill(Skill):
    name = "time"
    description = "Devuelve la hora y fecha actual en español"
    level = "SAFE"

    async def _do_execute(self, **params: object) -> SkillResult:
        now = datetime.now()

        dias: dict[int, str] = {
            0: "lunes",
            1: "martes",
            2: "miércoles",
            3: "jueves",
            4: "viernes",
            5: "sábado",
            6: "domingo",
        }
        meses: dict[int, str] = {
            1: "enero",
            2: "febrero",
            3: "marzo",
            4: "abril",
            5: "mayo",
            6: "junio",
            7: "julio",
            8: "agosto",
            9: "septiembre",
            10: "octubre",
            11: "noviembre",
            12: "diciembre",
        }

        dia_semana = dias[now.weekday()]
        dia = now.day
        mes = meses[now.month]
        año = now.year
        hora = now.strftime("%H:%M:%S")

        return SkillResult(
            success=True,
            data={
                "hora": hora,
                "fecha": f"{dia_semana}, {dia} de {mes} de {año}",
                "iso": now.isoformat(),
            },
        )
