"""TAD-LoRA's alternation policy: which factor group trains, and when it switches.

Pure functions of a few integers: nothing here touches a model, a payload or the
runtime, so the schedule can be tested without a model and checkpointed as a
handful of fields. Which parameters form a group is ``lora/factor_groups.py``'s
concern.

``B @ A`` is bilinear: non-convex jointly, convex in each factor with the other
held fixed. Updating one group at a time is alternating minimisation, and the
frozen group needs no gradient memory during a local episode.

With ``topology_aware`` the interval stretches when recent connectivity is poor.
A peer exports its local progress only when a mixing round reaches it, so a peer
that mixes rarely and alternates quickly could switch away from a factor before
any neighbour has seen its work on it. A well-connected peer keeps the base
interval. Connectivity can only lengthen the interval: never below the
configured base, and never beyond ``max_interval``.
"""
import math
from typing import Mapping, Optional

from federatedscope.contrib.decentralized.lora.factor_groups import (
    FACTOR_A,
    FACTOR_GROUPS,
    other_group,
)


class AlternationError(ValueError):
    """The alternation schedule was given something it cannot honour."""


def effective_interval(base_interval: int, *, topology_aware: bool,
                       recent_degree: Optional[int], window: int,
                       max_interval: int) -> int:
    """How many local episodes this peer holds one factor group for.

    ``recent_degree`` is how many offered edges named this peer within the
    topology window. ``None`` means no topology history is available (a peer
    built without one) and is treated as no reason to stretch, which keeps the
    schedule identical to the non-aware one.
    """
    if base_interval < 1:
        raise AlternationError(
            f"the alternation interval must be at least one local episode, "
            f"got {base_interval}")
    if max_interval < base_interval:
        raise AlternationError(
            f"max_interval {max_interval} is below the base interval "
            f"{base_interval}, so the ceiling would overrule the floor")
    if not topology_aware or recent_degree is None:
        return base_interval
    if window < 1:
        raise AlternationError(f"the topology window must be positive, got {window}")

    # One contact per window frame is "well connected" and stretches nothing.
    # Fewer contacts stretch proportionally, and zero contacts stretch to the
    # ceiling: a peer nobody has met has no reason to switch at all yet.
    contacts = max(0, int(recent_degree))
    if contacts >= window:
        return base_interval
    if contacts == 0:
        return max_interval
    stretched = int(math.ceil(base_interval * window / contacts))
    return max(base_interval, min(max_interval, stretched))


class AlternationSchedule:
    """TAD-LoRA's phase state: the active group, the episodes spent in it and the
    switch count, plus the interval settings.

    ``state_dict`` holds everything ``advance`` reads, so a resumed run continues
    the alternation where it stopped.
    """

    def __init__(self, base_interval: int, topology_aware: bool = True,
                 window: int = 16, max_interval: int = 8,
                 group: str = FACTOR_A, episodes_in_group: int = 0,
                 switches: int = 0):
        if group not in FACTOR_GROUPS:
            raise AlternationError(
                f"unknown starting factor group {group!r}; expected one of "
                f"{list(FACTOR_GROUPS)}")
        self.base_interval = int(base_interval)
        self.topology_aware = bool(topology_aware)
        self.window = int(window)
        self.max_interval = int(max_interval)
        self.group = group
        self.episodes_in_group = int(episodes_in_group)
        self.switches = int(switches)
        # Validate the configured pair once, here, rather than at the first switch.
        effective_interval(self.base_interval, topology_aware=self.topology_aware,
                           recent_degree=None, window=self.window,
                           max_interval=self.max_interval)

    def interval_now(self, recent_degree: Optional[int]) -> int:
        return effective_interval(
            self.base_interval, topology_aware=self.topology_aware,
            recent_degree=recent_degree, window=self.window,
            max_interval=self.max_interval)

    def advance(self, recent_degree: Optional[int] = None) -> Mapping[str, object]:
        """Count one completed local episode; switch group when the interval is up.

        Returns the group now active, whether it switched, the interval used,
        the episodes spent in the group and the switch count.
        """
        interval = self.interval_now(recent_degree)
        self.episodes_in_group += 1
        switched = False
        if self.episodes_in_group >= interval:
            self.group = other_group(self.group)
            self.episodes_in_group = 0
            self.switches += 1
            switched = True
        return {"group": self.group, "switched": switched,
                "interval": interval, "episodes_in_group": self.episodes_in_group,
                "switches": self.switches}

    def state_dict(self) -> Mapping[str, object]:
        return {"group": self.group, "episodes_in_group": self.episodes_in_group,
                "switches": self.switches, "base_interval": self.base_interval,
                "topology_aware": self.topology_aware, "window": self.window,
                "max_interval": self.max_interval}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        group = state["group"]
        if group not in FACTOR_GROUPS:
            raise AlternationError(
                f"checkpoint names factor group {group!r}, which does not exist")
        self.group = group
        self.episodes_in_group = int(state["episodes_in_group"])
        self.switches = int(state["switches"])
        self.base_interval = int(state["base_interval"])
        self.topology_aware = bool(state["topology_aware"])
        self.window = int(state["window"])
        self.max_interval = int(state["max_interval"])
