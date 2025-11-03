"""
Week 3: Composite Scoring Function

Combines multiple signals to score operation applicability:
1. Structural signal: Prefer shallow solutions
2. Learned signal: K_P weight for (operation, context)
3. Temporal signal: Avoid recently used operations (prevent loops)
4. Historical signal: Bayesian success rate

Formula: W(operation, context, depth) = combine(struct, learned, temporal, historical)
"""

from typing import Set, TYPE_CHECKING

if TYPE_CHECKING:
    from cig_poc.kp.store import KPStore


def composite_score(
    operation: str,
    context: str,
    depth: int,
    K_P: "KPStore",
    recently_used: Set[str],
    max_depth: int = 5,
    use_weighted_sum: bool = False
) -> float:
    """
    Composite scoring function: W(e, x, t)

    Combines multiple signals to score operation applicability.

    Args:
        operation: Operation name from K_R
        context: Context key (e.g., "small_prime", "short_list")
        depth: Current search depth
        K_P: Learned weight store
        recently_used: Set of operations already used in this path
        max_depth: Maximum allowed depth (hard cutoff)
        use_weighted_sum: If True, use weighted sum; else use multiplication

    Returns:
        Score in [0, 1], higher is better

    Scoring Strategies:
        - Multiplication (default): Simple, natural veto behavior, good for depth 1-3
        - Weighted sum: Better for deep searches, more stable scores

    Switch to weighted sum if:
        - Beam search rarely finds solutions at depth > 3
        - Scores all drop below 0.1 even for good paths
        - Learning stalls because all operations have similar (low) scores
    """

    # 1. Structural weight: prefer shallow depth
    struct_weight = 1.0 / (1.0 + depth)
    if depth >= max_depth:
        return 0.0  # Hard cutoff at max depth

    # 2. Learned weight from K_P
    learned_weight = K_P.get_weight(operation, context)

    # 3. Temporal weight: discourage loops
    temporal_weight = 1.0
    if operation in recently_used:
        temporal_weight = 0.5  # Penalize but don't eliminate

    # 4. Historical weight: Bayesian success rate
    success_rate = K_P.get_success_rate(operation, context)
    if success_rate is None:
        historical_weight = 0.5  # No data yet, neutral prior
    else:
        historical_weight = max(0.1, success_rate)  # Clamp to avoid zeros

    # Two combination strategies:
    if use_weighted_sum:
        # OPTION A: Weighted sum (better for deep searches)
        # Weights sum to 1.0
        alpha = 0.3   # Structural importance
        beta = 0.4    # Learned importance (highest - core of K_P)
        gamma = 0.1   # Temporal importance (lowest - just prevents loops)
        delta = 0.2   # Historical importance

        final_score = (
            alpha * struct_weight +
            beta * learned_weight +
            gamma * temporal_weight +
            delta * historical_weight
        )
    else:
        # OPTION B: Multiplication (default, simpler)
        # Issue: scores decay exponentially with depth
        # Example at depth=4: 0.2 * 0.5 * 0.5 * 0.5 = 0.025
        final_score = struct_weight * learned_weight * temporal_weight * historical_weight

    return final_score


def score_distribution_stats(scores: list) -> dict:
    """
    Analyze score distribution to help decide between multiplication and weighted sum.

    Args:
        scores: List of scores from beam search

    Returns:
        Statistics dictionary with mean, min, max, etc.
    """
    if not scores:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

    return {
        "mean": sum(scores) / len(scores),
        "min": min(scores),
        "max": max(scores),
        "count": len(scores),
        "below_0.05": sum(1 for s in scores if s < 0.05),
        "below_0.1": sum(1 for s in scores if s < 0.1)
    }


if __name__ == "__main__":
    # Test composite scoring
    print("=" * 60)
    print("Composite Scoring Tests")
    print("=" * 60)

    # Import K_P store
    import sys
    sys.path.insert(0, "/home/jay/ag0/cig/src")
    from cig_poc.kp.store import KPStore
    from cig_poc.kr.registry import OPERATIONS
    from cig_poc.kr.context import CONTEXT_TAXONOMY

    # Initialize K_P
    kp = KPStore()
    kp.initialize_uniform(OPERATIONS, CONTEXT_TAXONOMY)

    # Simulate some learning
    kp.update_weight("factors", "small_prime", True)
    kp.update_weight("factors", "small_prime", True)
    kp.update_weight("sum_list", "short_list", True)

    # Test scoring at different depths
    print("\n📊 Scores at different depths (Multiplication):")
    for depth in range(5):
        score = composite_score(
            "factors",
            "small_prime",
            depth,
            kp,
            set(),
            use_weighted_sum=False
        )
        print(f"  Depth {depth}: {score:.4f}")

    print("\n📊 Scores at different depths (Weighted Sum):")
    for depth in range(5):
        score = composite_score(
            "factors",
            "small_prime",
            depth,
            kp,
            set(),
            use_weighted_sum=True
        )
        print(f"  Depth {depth}: {score:.4f}")

    # Test with recently used operations
    print("\n🔄 Effect of recently used operations:")
    recently_used = set()
    score1 = composite_score("factors", "small_prime", 0, kp, recently_used)
    recently_used.add("factors")
    score2 = composite_score("factors", "small_prime", 0, kp, recently_used)
    print(f"  First use: {score1:.4f}")
    print(f"  Repeated use: {score2:.4f}")
    print(f"  Penalty: {(score1 - score2) / score1 * 100:.1f}%")

    # Compare multiplication vs weighted sum at depth=4
    print("\n⚖️  Multiplication vs Weighted Sum at depth=4:")
    score_mult = composite_score("factors", "small_prime", 4, kp, set(), use_weighted_sum=False)
    score_wsum = composite_score("factors", "small_prime", 4, kp, set(), use_weighted_sum=True)
    print(f"  Multiplication: {score_mult:.4f}")
    print(f"  Weighted Sum: {score_wsum:.4f}")
    print(f"  Ratio: {score_wsum / score_mult if score_mult > 0 else float('inf'):.2f}x higher")
