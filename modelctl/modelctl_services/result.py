"""The one result type every service operation returns.

Before this, each service had its own result dataclass with its own
subset of fields, so a caller handling several services had to know five
shapes and none of them could report a partial rollback. `ServiceResult`
is the common contract; the per-domain results subclass it and add their
own payload fields, so existing callers keep working while every result
gains the same envelope.

Fields:
    ok                 the operation achieved what it was asked to do
    messages           user-facing narration, in order
    warnings           things that went differently than asked but did not
                       fail the operation
    data               structured payload for programmatic consumers
    changed_resources  what this operation mutated, so a UI knows what to
                       refresh and an audit can say what moved
    job_id             the background job, when the work was submitted
                       rather than performed inline
    rollback_status    "" when nothing needed undoing, otherwise
                       "rolled_back", "partial", or "failed"
"""
from dataclasses import dataclass, field

ROLLBACK_NONE = ""
ROLLBACK_DONE = "rolled_back"
ROLLBACK_PARTIAL = "partial"
ROLLBACK_FAILED = "failed"


@dataclass
class ServiceResult:
    """Common envelope for every service operation."""
    ok: bool = True
    messages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    data: dict = field(default_factory=dict)
    changed_resources: list = field(default_factory=list)
    job_id: str | None = None
    rollback_status: str = ROLLBACK_NONE

    # --- construction helpers -------------------------------------------

    @classmethod
    def failure(cls, message, **kw):
        """A failed operation that changed nothing."""
        return cls(ok=False, messages=[message], **kw)

    def add_message(self, text):
        self.messages.append(text)
        return self

    def add_warning(self, text):
        self.warnings.append(text)
        return self

    def changed(self, *resources):
        """Record mutated resources as "kind:id" strings.

        Duplicates are collapsed: an operation that saves a profile twice
        during one call changed one profile, not two.
        """
        for r in resources:
            if r not in self.changed_resources:
                self.changed_resources.append(r)
        return self

    def as_dict(self):
        """JSON-safe view, for API responses and job payloads."""
        return {
            "ok": self.ok,
            "messages": list(self.messages),
            "warnings": list(self.warnings),
            "data": dict(self.data),
            "changed_resources": list(self.changed_resources),
            "job_id": self.job_id,
            "rollback_status": self.rollback_status,
        }
