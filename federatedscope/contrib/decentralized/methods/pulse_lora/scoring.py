"""Effective-update disagreement (the exact oracle and its sketch) and the
local-update magnitude.

Component ``p`` adds the rank-one product ``s * b_p (a_p)^T`` to the effective
weight update of each injected module ``m``, where ``s`` is the fixed LoRA
scale. The disagreement score of two peers ``i`` and ``j`` is

    s * sqrt( sum_m || b_i^m (a_i^m)^T - b_j^m (a_j^m)^T ||_F^2 )

which ranks components the same way as the sum of squared distances. The
Frobenius norm of a difference of rank-one matrices needs only inner products
of the factor vectors::

    || b_i a_i^T - b_j a_j^T ||_F^2
        = ||b_i||^2 ||a_i||^2 + ||b_j||^2 ||a_j||^2 - 2 (b_i . b_j)(a_i . a_j)

so no outer product is ever formed (for a 768x768 module, a few inner products
instead of a 590k-element matrix).

Each peer knows its own ``||a||^2`` and ``||b||^2``; only the terms that mix
the two peers' vectors need the partner's values. The exact score reads the
partner's tensors, which only the single-process simulation can do, so it is an
oracle. The sketch estimates those terms from shared deterministic Rademacher
projections ``P_m a`` and ``Q_m b``, whose inner products are unbiased
estimates of the true ones; each peer publishes its projections in the
manifest, where they are charged as control bytes.
"""
import hashlib
import math
from typing import Dict, List, Mapping, Sequence, Tuple

import torch

from federatedscope.contrib.decentralized.methods.pulse_lora.components import (
    SIDE_A, SIDE_B, CanonicalComponentSchema,
)


class ScoringError(ValueError):
    pass


# -- exact: local norms and the oracle score ------------------------------------

def component_norms(schema: CanonicalComponentSchema,
                    block: Mapping[str, torch.Tensor]) -> List[Tuple[float, float]]:
    """Per-module ``(||a||^2, ||b||^2)`` - exact, local, two floats each."""
    out = []
    for spec in schema.lora.modules:
        a = block[f"{spec.module}|{SIDE_A}"].detach().to(torch.float64)
        b = block[f"{spec.module}|{SIDE_B}"].detach().to(torch.float64)
        out.append((float(a.dot(a)), float(b.dot(b))))
    return out


def exact_disagreement(schema: CanonicalComponentSchema,
                       local: Mapping[str, torch.Tensor],
                       remote: Mapping[str, torch.Tensor],
                       report: dict = None) -> float:
    """The oracle score. Reads both peers' actual tensors.

    Computed through the difference, not through the expansion in the module
    docstring, which subtracts two nearly equal large numbers when the peers are
    close (the usual case, since gossip keeps them close) and loses precision.
    With ``b_j = b_i + db`` and ``a_j = a_i + da``,

        ||b_i a_i^T - b_j a_j^T||_F^2
            = ||b_i||^2 ||da||^2 + ||db||^2 ||a_j||^2 + 2 (b_i.db)(da.a_j)

    where every term is second order in the difference. Same cost, six inner
    products per module, in float64.

    The value is symmetric in the two peers only in exact arithmetic, so the session
    policy passes the two blocks in the same order on both endpoints (the lower peer
    id first) to get bit-identical scores.

    Pass ``report`` to receive the total before clamping; see :func:`_finish`.
    """
    total = 0.0
    for spec in schema.lora.modules:
        ai = local[f"{spec.module}|{SIDE_A}"].detach().to(torch.float64)
        bi = local[f"{spec.module}|{SIDE_B}"].detach().to(torch.float64)
        aj = remote[f"{spec.module}|{SIDE_A}"].detach().to(torch.float64)
        bj = remote[f"{spec.module}|{SIDE_B}"].detach().to(torch.float64)
        da, db = aj - ai, bj - bi
        total += (float(bi.dot(bi)) * float(da.dot(da))
                  + float(db.dot(db)) * float(aj.dot(aj))
                  + 2.0 * float(bi.dot(db)) * float(da.dot(aj)))
    return _finish(schema, total, report)


def _finish(schema: CanonicalComponentSchema, total: float, report: dict = None) -> float:
    """``scale * sqrt(max(0, total))``, recording a clamp in ``report``.

    A negative total cannot occur in exact arithmetic; it comes from rounding
    or, for the sketch, from estimation noise. A clamped score ties at 0.0 with
    two identical blocks and is then ordered by component id, so ``report``
    receives the raw total and a ``clamped`` flag for the caller to count.
    """
    if report is not None:
        report["raw_total"] = total
        report["clamped"] = total < 0.0
    return float(schema.scale) * math.sqrt(max(0.0, total))


# -- deployable sketch -----------------------------------------------------------

def projection(schema: CanonicalComponentSchema, module: str, side: str,
               length: int, dim: int) -> torch.Tensor:
    """A shared deterministic Rademacher projection of shape ``(dim, length)``.

    Entries are ``+-1/sqrt(dim)``, from a generator seeded by a hash of the
    structural hash, the component seed, the module, the side, the length and
    the sketch width, so every peer builds the same matrix without exchanging
    it and without drawing from any shared random stream.
    """
    material = (f"{schema.lora.structural_hash}:{schema.seed}:sketch:{module}:"
                f"{side}:{length}:{dim}")
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    generator = torch.Generator()
    generator.manual_seed(int.from_bytes(digest[:8], "big") % (2 ** 63 - 1))
    signs = torch.randint(0, 2, (dim, length), generator=generator,
                          dtype=torch.int8) * 2 - 1
    return signs.to(torch.float64) / math.sqrt(dim)


def component_sketch(schema: CanonicalComponentSchema,
                     block: Mapping[str, torch.Tensor],
                     dim: int) -> List[Tuple[List[float], List[float]]]:
    """Per-module ``(P a, Q b)`` of one component's block, as the manifest
    carries them (charged as control bytes)."""
    if dim <= 0:
        raise ScoringError(f"sketch dim must be positive, got {dim}")
    out = []
    for spec in schema.lora.modules:
        a = block[f"{spec.module}|{SIDE_A}"].detach().to(torch.float64)
        b = block[f"{spec.module}|{SIDE_B}"].detach().to(torch.float64)
        pa = projection(schema, spec.module, SIDE_A, spec.in_features, dim) @ a
        qb = projection(schema, spec.module, SIDE_B, spec.out_features, dim) @ b
        out.append(([float(v) for v in pa], [float(v) for v in qb]))
    return out


def sketch_disagreement(schema: CanonicalComponentSchema,
                        local_norms: Sequence[Tuple[float, float]],
                        local_sketch: Sequence[Tuple[Sequence[float], Sequence[float]]],
                        remote_norms: Sequence[Tuple[float, float]],
                        remote_sketch: Sequence[Tuple[Sequence[float], Sequence[float]]],
                        report: dict = None) -> float:
    """The sketch estimate of :func:`exact_disagreement`, from manifest values.

    Uses the same difference identity. ``P(a_j - a_i) = P a_j - P a_i`` is
    computed exactly from the two published sketches, so the estimation noise
    scales with ``||da||``, the quantity being measured, rather than with
    ``||a||``, the component's magnitude. Estimating the cross terms of the
    expansion instead would bury the small differences between close peers in
    that noise.

    The squared norms of each peer's own factors are exact (published in the
    manifest); every quantity that involves both peers' vectors comes from the
    sketches.
    """
    count = len(schema.lora.modules)
    for name, seq in (("local_norms", local_norms), ("remote_norms", remote_norms),
                      ("local_sketch", local_sketch), ("remote_sketch", remote_sketch)):
        if len(seq) != count:
            raise ScoringError(
                f"{name} covers {len(seq)} modules but the schema has {count}")

    total = 0.0
    for index in range(count):
        aa_i, bb_i = local_norms[index]
        aa_j, bb_j = remote_norms[index]
        pa_i, qb_i = local_sketch[index]
        pa_j, qb_j = remote_sketch[index]
        if len(pa_i) != len(pa_j) or len(qb_i) != len(qb_j):
            raise ScoringError("sketch widths disagree between the two peers")

        pda = [x - y for x, y in zip(pa_j, pa_i)]
        qdb = [x - y for x, y in zip(qb_j, qb_i)]
        da_norm2 = sum(v * v for v in pda)
        db_norm2 = sum(v * v for v in qdb)

        # Each orientation of the identity is symmetric in the two peers only in
        # exact arithmetic, and the endpoints could then rank differently and
        # select different sets. The average of the two orientations gives the
        # same float at both endpoints.
        forward = (bb_i * da_norm2 + db_norm2 * aa_j
                   + 2.0 * sum(x * y for x, y in zip(qb_i, qdb))
                   * sum(x * y for x, y in zip(pda, pa_j)))
        reverse = (bb_j * da_norm2 + db_norm2 * aa_i
                   + 2.0 * sum(x * y for x, y in zip(qb_j, qdb))
                   * sum(x * y for x, y in zip(pda, pa_i)))
        total += 0.5 * (forward + reverse)
    return _finish(schema, total, report)


# -- deployable, purely local -------------------------------------------------------

def local_update_magnitude_for(schema: CanonicalComponentSchema, component: int,
                               block: Mapping[str, torch.Tensor],
                               report: dict = None) -> float:
    """The local-update magnitude: how far this peer has moved the component
    from its canonical initial value.

    :func:`exact_disagreement` against the deterministic initial value every
    peer derives, so it needs only local state. Under ``local_update``,
    ``shuffled_score`` and ``survivor_only`` each peer publishes it in the
    manifest for every materialized component. The component id is required
    because the block's keys name only the module and the side.
    """
    return exact_disagreement(schema, block,
                              schema.initial_component(int(component)), report)
