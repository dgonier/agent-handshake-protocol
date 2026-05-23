"""End-to-end demo: register a Neo4j KG resource, run a TeacherAgent.

Requires a running Neo4j (see ``docker-compose.neo4j.yml``) and the
``[kg]`` extra installed (``pip install -e ".[kg]"``).

Drives one ``teacher.judge`` interaction and prints the resulting graph
fragment back via a Cypher count. The in-memory backend works the same
way — swap the resource factory if you want to try it without Neo4j.
"""

from __future__ import annotations

import asyncio
import os

from ahp.adapters import ResourceRegistry, resource
from ahp.adapters.knowledge_graph import KG_KIND
from ahp.adapters.neo4j_kg import Neo4jKnowledgeGraph
from ahp.adapters.teacher_agent import Criterion, Judgement, Rubric, TeacherAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


JUDGE_RUBRIC = Rubric(
    name="financial-reasoning",
    description="how well an argument grounds equities claims in evidence",
    criteria=(
        Criterion("evidence", "cites concrete data points", weight=2.0),
        Criterion("clarity", "easy to follow", weight=1.0),
    ),
)


def keyword_judge(rubric: Rubric, body: object) -> dict[str, float]:
    """Toy scorer: rewards mentions of 'evidence' and short bodies."""
    text = str(body).lower()
    return {
        "evidence": 1.0 if "evidence" in text or "data" in text else 0.2,
        "clarity": 1.0 if len(text) < 200 else 0.5,
    }


async def main() -> None:
    # Standard AHP stack — real Redis here, fakeredis in tests.
    from ahp.core.compatibility import CompatibilityMatrix
    from ahp.engine.router import ProtocolEngine
    from ahp.engine.thread_manager import ThreadManager
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    import redis.asyncio as aioredis  # type: ignore

    redis_url = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")
    redis = aioredis.from_url(redis_url, decode_responses=True)
    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=30)
    cache = ProtocolCache(redis)
    threads = ThreadManager(redis, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads, default_timeout=5.0,
    )

    # Register a Neo4j KG resource. Any agent whose address matches
    # ``acme.*.finance.equities.*.*.*`` will receive this backend
    # via build_kg_backend.
    resources = ResourceRegistry()

    @resource(
        "acme", KG_KIND, "finance", "equities",
        name="primary",
        description="canonical equities belief graph",
        cleanup=lambda g: g.close(),
    )
    def make_primary_kg():
        return Neo4jKnowledgeGraph(
            vector_dimensions=1536,
            auto_create_vector_index=True,
        )

    teacher_address = AgentAddress.parse(
        "acme.teacher.finance.equities.s.session.judge1",
    )
    teacher = TeacherAgent(
        teacher_address,
        engine,
        rubric=JUDGE_RUBRIC,
        judge_fn=keyword_judge,
        kg_backend=resources.get(f"acme.{KG_KIND}.finance.equities.primary"),
    )
    await teacher.register()
    await teacher.start()

    student_address = AgentAddress.parse(
        "acme.adversarial.finance.equities.s.session.student1",
    )
    sample = Message(
        source=student_address,
        target=teacher_address,
        verb="SEND-GET",
        code=Code.TEACHER_JUDGE,
        thread="thread::demo::teacher",
        body="Tesla's recent earnings beat shows the data supports the bull case.",
    )

    response = await engine.handle(sample, timeout=10.0)
    if response is None:
        print("teacher did not respond within the timeout")
        return

    print("composite score:", response.body["composite"])
    print("rationale:      ", response.body["rationale"])
    print()
    print("KG node count:")
    for kind in ("Agent", "Rubric", "Judgement"):
        nodes = teacher.kg.list_nodes(kind=kind)
        print(f"  {kind:<12} {len(nodes)}")

    await teacher.stop()
    await teacher.deregister()
    await bus.close()
    await resources.close_all()


if __name__ == "__main__":
    asyncio.run(main())
