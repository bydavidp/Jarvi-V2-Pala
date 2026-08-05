import logging
from typing import Any

from security.policy import Policy
from skills.base import SkillRegistry, SkillResult

logger = logging.getLogger("jarvis.orchestrator")


class Orchestrator:
    def __init__(self, policy: Policy, registry: SkillRegistry) -> None:
        self.policy = policy
        self.registry = registry

    async def dispatch(
        self, skill_name: str, operation: str, params: dict[str, Any] | None = None
    ) -> SkillResult:
        logger.info("Despachando skill: %s.%s", skill_name, operation)
        return await self.registry.execute(skill_name, operation, params)
