"""Risk-based HITL and MCP tool firewall for policy-gated tool invocation."""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import structlog
from opentelemetry import trace

from ag2.hitl.context import Context
from ag2.providers.base import ToolSpec
from ag2.tools.tool import Tool, ToolError

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer("ag2.tools")

ToolDecision = Literal["allow", "confirm", "deny"]
ToolChoice = Literal["approve", "deny"]
ToolPredicate = Callable[[str, dict[str, Any]], bool]

_VALID_DECISIONS: tuple[str, ...] = ("allow", "confirm", "deny")
_MAX_PREVIEW_CHARS = 200


class ToolDeniedError(ToolError):
    """Raised when a guarded tool call is blocked by policy or rejected by a human."""

    def __init__(self, tool_name: str, *, decision: ToolDecision, reason: str = "") -> None:
        self.tool_name = tool_name
        self.decision = decision
        self.reason = reason
        detail = f": {reason}" if reason else ""
        super().__init__(f"Tool call {tool_name!r} denied ({decision}){detail}")


@dataclass(frozen=True, slots=True)
class ToolPolicyRule:
    """Single policy rule matching tool names and/or call arguments."""

    decision: ToolDecision
    pattern: str | None = None
    predicate: ToolPredicate | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.decision not in _VALID_DECISIONS:
            raise ValueError(
                f"Invalid tool decision {self.decision!r}; expected one of {_VALID_DECISIONS}"
            )
        if self.pattern is None and self.predicate is None:
            raise ValueError("ToolPolicyRule requires at least one of 'pattern' or 'predicate'")

    def matches(self, tool_name: str, arguments: dict[str, Any]) -> bool:
        """Return whether this rule matches the tool call."""
        if self.pattern is not None and not fnmatch.fnmatchcase(tool_name, self.pattern):
            return False
        if self.predicate is not None:
            return bool(self.predicate(tool_name, arguments))
        return True


@dataclass(frozen=True, slots=True)
class ToolVerdict:
    """Result of evaluating a ToolPolicy against a tool call."""

    decision: ToolDecision
    reason: str
    matched_rule: ToolPolicyRule | None = None


class ToolPolicy:
    """Ordered firewall policy mapping tool calls to allow, confirm, or deny decisions.

    Explicit ``rules`` are evaluated first in order. Pattern shorthands are then
    checked from strictest to least strict (``deny``, then ``confirm``, then
    ``allow``), falling back to ``default``.
    """

    def __init__(
        self,
        *,
        rules: Sequence[ToolPolicyRule] = (),
        allow: Sequence[str] = (),
        confirm: Sequence[str] = (),
        deny: Sequence[str] = (),
        default: ToolDecision = "allow",
        default_reason: str = "",
        raise_on_deny: bool = False,
        confirm_timeout: float | None = None,
        timeout_default: ToolChoice | None = "deny",
        raise_on_timeout: bool = False,
    ) -> None:
        if default not in _VALID_DECISIONS:
            raise ValueError(
                f"Invalid default decision {default!r}; expected one of {_VALID_DECISIONS}"
            )
        if timeout_default is not None and timeout_default not in ("approve", "deny"):
            raise ValueError(
                f"Invalid timeout_default {timeout_default!r}; expected 'approve', 'deny', or None"
            )

        tier_rules: list[ToolPolicyRule] = [
            *(
                ToolPolicyRule("deny", pattern=pat, reason="blocked by deny pattern")
                for pat in deny
            ),
            *(
                ToolPolicyRule("confirm", pattern=pat, reason="requires human approval")
                for pat in confirm
            ),
            *(ToolPolicyRule("allow", pattern=pat, reason="allowed by pattern") for pat in allow),
        ]
        self.rules: tuple[ToolPolicyRule, ...] = (*rules, *tier_rules)
        self.default: ToolDecision = default
        self.default_reason = default_reason or f"default policy ({default})"
        self.raise_on_deny = raise_on_deny
        self.confirm_timeout = confirm_timeout
        self.timeout_default: ToolChoice | None = timeout_default
        self.raise_on_timeout = raise_on_timeout

    def evaluate(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolVerdict:
        """Evaluate rules against ``tool_name`` and ``arguments``."""
        args = arguments if arguments is not None else {}
        for rule in self.rules:
            if rule.matches(tool_name, args):
                reason = rule.reason or f"matched rule ({rule.decision})"
                return ToolVerdict(decision=rule.decision, reason=reason, matched_rule=rule)
        return ToolVerdict(decision=self.default, reason=self.default_reason, matched_rule=None)

    def may_confirm(self, tool_name: str) -> bool:
        """Return whether ``tool_name`` could require HITL confirmation under this policy."""
        if self.default == "confirm":
            return True
        for rule in self.rules:
            if rule.decision != "confirm":
                continue
            if rule.pattern is None or fnmatch.fnmatchcase(tool_name, rule.pattern):
                return True
        return False


def _format_args_preview(arguments: dict[str, Any]) -> str:
    try:
        rendered = json.dumps(arguments, sort_keys=True, default=str)
    except Exception:
        rendered = str(arguments)
    if len(rendered) > _MAX_PREVIEW_CHARS:
        return f"{rendered[: _MAX_PREVIEW_CHARS - 3]}..."
    return rendered


class GuardedTool(Tool):
    """Tool wrapper enforcing ``ToolPolicy`` decisions and HITL confirmation before execution."""

    def __init__(self, inner: Tool, policy: ToolPolicy) -> None:
        self.inner = inner
        self.policy = policy
        self._fn = inner._fn
        self.name = inner.name
        self.description = inner.description
        self.is_async = True
        self._params_model = inner._params_model
        self._context_param = inner._context_param
        self._param_names = inner._param_names
        self._required_keys = inner._required_keys
        self.parameters = dict(inner.parameters)

    @property
    def expects_context(self) -> bool:
        """Return True if the wrapped tool or firewall policy may request HITL input."""
        return self.inner.expects_context or self.policy.may_confirm(self.name)

    @property
    def spec(self) -> ToolSpec:
        """Preserve the exact JSON schema and ToolSpec of the wrapped tool."""
        return self.inner.spec

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Delegate argument validation directly to the wrapped tool."""
        return self.inner.validate_arguments(arguments)

    def _reject(self, actor: str, verdict: ToolVerdict) -> str:
        if self.policy.raise_on_deny:
            raise ToolDeniedError(self.name, decision="deny", reason=verdict.reason)
        detail = f": {verdict.reason}" if verdict.reason else ""
        return f"Tool call denied by {actor} for {self.name!r}{detail}"

    async def _request_confirmation(
        self,
        *,
        verdict: ToolVerdict,
        args_preview: str,
        context: Context | None,
    ) -> ToolChoice:
        effective_ctx = context if context is not None else Context()
        reason_suffix = f" ({verdict.reason})" if verdict.reason else ""
        prompt = f"Approve execution of tool '{self.name}' with args {args_preview}{reason_suffix}?"
        choice = await effective_ctx.input(
            prompt,
            timeout=self.policy.confirm_timeout,
            choices=("approve", "deny"),
            default=self.policy.timeout_default,
            raise_on_timeout=self.policy.raise_on_timeout,
        )
        return cast(ToolChoice, choice)

    async def invoke(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        context: Context | None = None,
    ) -> Any:
        """Validate arguments, enforce policy/HITL checks, and invoke the wrapped tool."""
        raw_args = dict(arguments) if arguments is not None else {}
        self.inner.validate_arguments(raw_args)

        verdict = self.policy.evaluate(self.name, raw_args)
        args_preview = _format_args_preview(raw_args)

        with tracer.start_as_current_span(
            "ag2.tool.guard",
            attributes={
                "ag2.tool.name": self.name,
                "ag2.tool.policy.decision": verdict.decision,
                "ag2.tool.policy.reason": verdict.reason,
            },
        ) as span:
            if verdict.decision == "deny":
                span.set_attribute("ag2.tool.guard.outcome", "denied_by_policy")
                logger.warning(
                    "ag2.tool.guard.denied",
                    tool=self.name,
                    reason=verdict.reason,
                )
                return self._reject("policy", verdict)

            if verdict.decision == "confirm":
                choice = await self._request_confirmation(
                    verdict=verdict,
                    args_preview=args_preview,
                    context=context,
                )
                if choice == "deny":
                    span.set_attribute("ag2.tool.guard.outcome", "denied_by_human")
                    logger.info(
                        "ag2.tool.guard.human_denied",
                        tool=self.name,
                        reason=verdict.reason,
                    )
                    return self._reject("human reviewer", verdict)

                span.set_attribute("ag2.tool.guard.outcome", "approved_by_human")
                logger.info(
                    "ag2.tool.guard.human_approved",
                    tool=self.name,
                    reason=verdict.reason,
                )
            else:
                span.set_attribute("ag2.tool.guard.outcome", "allowed")
                logger.debug(
                    "ag2.tool.guard.allowed",
                    tool=self.name,
                    reason=verdict.reason,
                )

        return await self.inner.invoke(raw_args, context=context)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate synchronous direct calls to the wrapped tool."""
        return self.inner(*args, **kwargs)

    def __repr__(self) -> str:
        return f"GuardedTool(name={self.name!r}, default={self.policy.default!r})"


def guard_tool(target: Tool, policy: ToolPolicy) -> GuardedTool:
    """Wrap a native or MCP ``Tool`` with a ``ToolPolicy`` firewall."""
    return GuardedTool(target, policy)


def guard_tools(tools: Iterable[Tool], policy: ToolPolicy) -> list[Tool]:
    """Wrap every tool in ``tools`` with ``policy`` while preserving tool order."""
    return [GuardedTool(item, policy) for item in tools]
