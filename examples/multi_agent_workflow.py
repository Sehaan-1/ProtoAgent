"""Multi-agent workflow example — Hub + WorkflowChannel with TransitionGraph.

Run:
    OPENAI_API_KEY=sk-... uv run python examples/multi_agent_workflow.py

Three agents are wired into a linear pipeline:
    drafter -> reviewer -> finaliser -> END
"""

from __future__ import annotations

import asyncio

from ag2 import (
    END,
    Agent,
    Hub,
    OpenAICompletionsConfig,
    TransitionGraph,
    workflow,
)


def _make_agent(name: str, system_message: str) -> Agent:
    """Build an agent backed by OpenAI gpt-4o-mini."""
    return Agent(
        name=name,
        system_message=system_message,
        model=OpenAICompletionsConfig(model="gpt-4o-mini"),
    )


async def main() -> None:
    """Run a three-step drafting pipeline inside a Hub."""
    drafter = _make_agent("drafter", "Draft a short paragraph on the given topic.")
    reviewer = _make_agent("reviewer", "Review the paragraph and suggest one improvement.")
    finaliser = _make_agent(
        "finaliser", "Apply the reviewer suggestion and output the final polished paragraph."
    )

    graph = (
        TransitionGraph()
        .set_entry("drafter")
        .add_edge("drafter", "reviewer")
        .add_edge("reviewer", "finaliser")
        .add_edge("finaliser", END)
    )

    async with Hub(close_agents=True) as hub:
        hub.register(drafter)
        hub.register(reviewer)
        hub.register(finaliser)

        result = await workflow(
            hub,
            graph,
            "Write about the benefits of async programming in Python.",
        )
        print("=== Final output ===")
        print(result.final_text)


if __name__ == "__main__":
    asyncio.run(main())
