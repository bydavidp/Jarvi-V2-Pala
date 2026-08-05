from skills.base import SkillRegistry
from skills.time_skill import TimeSkill


def discover_skills(registry: SkillRegistry) -> None:
    registry.register(TimeSkill())
