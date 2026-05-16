"""Agent-wallet provisioning + teardown.

Every agent is a wallet owner. At construction time it gets seeded
with a small starting balance so it can dispatch a handful of calls
before having to earn. At deregistration time the residual is
returned according to the agent's lifecycle:

* ``session`` agents refund to their host server's wallet
* ``longterm`` agents persist their balance (the address survives
  for the next time it comes online)
* ``ephemeral`` agents return residual to the commons pool

Funding sources:

* ``session`` agents: funded from the host server's wallet
* ``longterm`` agents: funded from the commons pool
* ``ephemeral`` agents: funded from the commons pool

The defaults are tuned so a fresh ``session`` agent can dispatch
~10-20 small-tier calls before earning. Concrete numbers below.
"""

from __future__ import annotations

from typing import Final

from ahp.core.address import AgentAddress
from ahp.economy.wallet import (
    INITIAL_FUND,
    InsufficientFundsError,
    Wallet,
)


# Wallet owner strings for the protocol-level pools.
COMMONS_WALLET: Final[str] = "__commons__"
BROKER_WALLET: Final[str] = "__broker__"


# Lifecycle-keyed starting funds in credits. A small-tier 500-char
# call costs ~0.20 at neutral multipliers, so 5 credits = ~25 calls
# before needing to earn.
STARTING_FUND_BY_LIFECYCLE: Final[dict[str, float]] = {
    "session":   5.0,    # ~25 small-tier calls
    "longterm":  15.0,   # bootstrap over a longer window
    "ephemeral": 1.0,    # one-shot agents shouldn't have much runway
    "stale-ok":  3.0,    # similar shape to session
}


def funding_source_for(lifecycle: str, host_server_id: str | None) -> str:
    """Return the wallet owner that funds an agent of this lifecycle.

    Sessions are funded by the host server (the entity that decided
    to spawn the agent for its own workflow). Long-lived and
    ephemeral agents are funded by the commons pool — they're more
    like shared community resources or one-shot bursts that no
    single server should pay for.
    """
    if lifecycle == "session" and host_server_id:
        return host_server_id
    return COMMONS_WALLET


def refund_destination_for(
    lifecycle: str,
    host_server_id: str | None,
) -> str | None:
    """Where residual credits flow on agent deregister.

    * ``session`` → host server (mirror of funding)
    * ``longterm`` → ``None`` (balance persists in the agent's wallet)
    * ``ephemeral`` → commons
    * ``stale-ok`` → host server
    """
    if lifecycle == "session":
        return host_server_id
    if lifecycle == "longterm":
        return None
    if lifecycle == "ephemeral":
        return COMMONS_WALLET
    if lifecycle == "stale-ok":
        return host_server_id
    return host_server_id  # conservative default


class AgentBanker:
    """High-level helper: fund agents on provisioning, refund on teardown.

    Talks to two wallets per operation (funder/agent on provisioning;
    agent/destination on teardown). Each wallet operation is its own
    Redis transaction; the banker doesn't try to make the pair
    atomic because either half failing is recoverable (a stranded
    balance can be swept later by a janitor).
    """

    def __init__(self, redis_client) -> None:
        self._redis = redis_client

    def _wallet(self, owner: str) -> Wallet:
        return Wallet(self._redis, owner=owner)

    async def fund_agent(
        self,
        address: AgentAddress,
        *,
        host_server_id: str | None,
        amount: float | None = None,
    ) -> float:
        """Seed an agent's wallet at provisioning time.

        Returns the amount actually credited. Raises
        :class:`InsufficientFundsError` if the funder doesn't have
        the balance — in which case the caller should fail
        provisioning rather than spawn an unfunded agent.
        """
        if amount is None:
            amount = STARTING_FUND_BY_LIFECYCLE.get(
                address.lifecycle, INITIAL_FUND,
            )
        if amount <= 0:
            return 0.0

        funder = funding_source_for(address.lifecycle, host_server_id)
        funder_w = self._wallet(funder)
        agent_w = self._wallet(str(address))

        # Withdraw from funder. If they don't have it, fail loudly.
        funder_state = await funder_w.get_state()
        if funder_state.available < amount:
            raise InsufficientFundsError(
                f"funder {funder!r} cannot seed agent {address}: "
                f"available {funder_state.available:.4f} < required {amount:.4f}"
            )

        # Two writes: debit funder, credit agent. Not atomic across
        # both, but each is atomic on its own wallet. A stranded
        # debit (without matching credit) is recoverable from the
        # funder's history.
        hold_id = f"fund:{address}"
        await funder_w.hold(
            hold_id=hold_id, amount=amount,
            reason=f"fund agent {address}",
        )
        await funder_w.settle_against_hold(
            hold_id=hold_id, debit=amount,
            reason=f"funded agent {address}",
        )
        await agent_w.topup(amount, reason=f"funded by {funder}")
        return amount

    async def drain_agent(
        self,
        address: AgentAddress,
        *,
        host_server_id: str | None,
    ) -> float:
        """Refund an agent's residual balance at teardown.

        Returns the amount transferred. For ``longterm`` agents,
        returns 0 and leaves the balance in place.
        """
        destination = refund_destination_for(
            address.lifecycle, host_server_id,
        )
        if destination is None:
            return 0.0  # longterm: balance stays

        agent_w = self._wallet(str(address))
        state = await agent_w.get_state()
        residual = state.balance
        if residual <= 0:
            return 0.0

        # Drain the agent's balance into the destination wallet.
        # We do this as a topup on the destination + a manual debit
        # on the agent. The agent's wallet should end at 0.
        await self._wallet(destination).topup(
            residual,
            reason=f"residual from agent {address}",
        )
        # Now zero out the agent by holding the full balance and
        # settling it to nothing.
        hold_id = f"drain:{address}"
        await agent_w.hold(
            hold_id=hold_id, amount=residual,
            reason=f"drain to {destination}",
        )
        await agent_w.settle_against_hold(
            hold_id=hold_id, debit=residual,
            reason=f"drained to {destination}",
        )
        return residual
