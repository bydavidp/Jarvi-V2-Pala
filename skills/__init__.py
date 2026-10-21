from typing import TYPE_CHECKING

from skills.base import SkillRegistry
from skills.code import CodeSkill
from skills.time_skill import TimeSkill

if TYPE_CHECKING:
    from core.config import Config


def discover_skills(registry: SkillRegistry, config: "Config | None" = None) -> None:
    """Registra las skills. Con ``config`` se habilitan las que necesitan ajustes."""
    registry.register(TimeSkill())

    if config is not None:
        code_cfg = config.code
        registry.register(
            CodeSkill(
                roots=code_cfg.roots,
                max_file_bytes=code_cfg.max_file_bytes,
                blocked_extensions=code_cfg.blocked_extensions or None,
                vs_code_command=code_cfg.vs_code_command,
            )
        )
