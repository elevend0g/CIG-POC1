"""
Week 2: K_P - Learned Heuristic Weights

Pattern Knowledge store that learns operation selection weights over time.
Structure: Dict[(operation_name, context_key), weight]

This enables the system to learn which operations work better in different contexts
without modifying the core K_R operations.
"""

from typing import Dict, Tuple, List, Optional
import json
import os


class KPStore:
    """
    Pattern Knowledge: Learned weights for operation selection.

    Structure: Dict[(operation_name, context_key), weight]

    Example:
        K_P[("factors", "small_prime")] = 0.95
        K_P[("factors", "large_composite")] = 0.60
        K_P[("sum_list", "short_list")] = 0.98

    The system learns these weights through temporal difference learning,
    adjusting based on whether operations lead to successful solutions.
    """

    def __init__(self):
        """Initialize empty K_P store"""
        self.weights: Dict[Tuple[str, str], float] = {}
        self.success_counts: Dict[Tuple[str, str], int] = {}
        self.attempt_counts: Dict[Tuple[str, str], int] = {}

    def initialize_uniform(self, operations: Dict, contexts: Dict):
        """
        Start with uniform weights (0.5) for all operation-context combinations.

        Args:
            operations: Dictionary of operation names from K_R
            contexts: Dictionary mapping operation names to list of context keys

        Example:
            >>> kp = KPStore()
            >>> kp.initialize_uniform(
            ...     {"factors": op, "sum_list": op2},
            ...     {"factors": ["small_prime", "small_composite"], "sum_list": ["short_list"]}
            ... )
        """
        for op_name in operations:
            # Get contexts for this operation, default to ["generic"]
            op_contexts = contexts.get(op_name, ["generic"])

            for context in op_contexts:
                key = (op_name, context)
                self.weights[key] = 0.5
                self.success_counts[key] = 0
                self.attempt_counts[key] = 0

        print(f"✅ K_P initialized with {len(self.weights)} (operation, context) pairs")

    def get_weight(self, operation: str, context: str) -> float:
        """
        Retrieve learned weight for (operation, context) pair.

        Args:
            operation: Operation name
            context: Context key

        Returns:
            Weight value in [0, 1], default 0.5 if not found
        """
        key = (operation, context)
        return self.weights.get(key, 0.5)  # Default: 0.5

    def update_weight(self, operation: str, context: str, success: bool, alpha: float = 0.1):
        """
        Update K_P based on outcome using temporal difference learning.

        Formula: new_weight = old_weight + α * (reward - old_weight)

        If operation succeeded: reward = +1.0
        If operation failed: reward = -0.5

        Args:
            operation: Operation name
            context: Context key
            success: Whether the operation led to a successful solution
            alpha: Learning rate (default 0.1)
        """
        key = (operation, context)
        old_weight = self.get_weight(operation, context)

        reward = 1.0 if success else -0.5
        new_weight = old_weight + alpha * (reward - old_weight)

        # Clamp to [0, 1]
        self.weights[key] = max(0.0, min(1.0, new_weight))

        # Track statistics
        if key not in self.attempt_counts:
            self.attempt_counts[key] = 0
            self.success_counts[key] = 0

        self.attempt_counts[key] += 1
        if success:
            self.success_counts[key] += 1

    def get_success_rate(self, operation: str, context: str) -> Optional[float]:
        """
        Empirical success rate for (operation, context) pair.

        This provides a Bayesian prior for the composite scoring function.

        Args:
            operation: Operation name
            context: Context key

        Returns:
            Success rate in [0, 1], or None if no attempts yet
        """
        key = (operation, context)
        attempts = self.attempt_counts.get(key, 0)
        successes = self.success_counts.get(key, 0)

        if attempts == 0:
            return None  # No data yet

        return successes / attempts

    def get_statistics(self) -> Dict:
        """
        Get overall statistics about K_P learning.

        Returns:
            Dictionary with:
                - total_pairs: Number of (op, context) pairs tracked
                - total_attempts: Total learning updates
                - pairs_with_data: Number of pairs that have been tried
                - top_performers: Top 10 (op, context) by success rate
                - worst_performers: Bottom 10 (op, context) by success rate
        """
        total_attempts = sum(self.attempt_counts.values())
        pairs_with_data = sum(1 for v in self.attempt_counts.values() if v > 0)

        # Get pairs sorted by success rate
        pairs_with_attempts = [
            (k, self.get_success_rate(k[0], k[1]), self.attempt_counts[k])
            for k in self.weights.keys()
            if self.attempt_counts.get(k, 0) > 0
        ]

        # Sort by success rate (descending)
        pairs_with_attempts.sort(key=lambda x: (x[1] or 0, x[2]), reverse=True)

        return {
            "total_pairs": len(self.weights),
            "total_attempts": total_attempts,
            "pairs_with_data": pairs_with_data,
            "top_performers": pairs_with_attempts[:10],
            "worst_performers": pairs_with_attempts[-10:] if len(pairs_with_attempts) > 10 else []
        }

    def save_to_json(self, filepath: str):
        """
        Persist learned weights for analysis.

        Saves two versions of data:
            1. weights: Current learned weights
            2. success_rates: Empirical success rates

        Args:
            filepath: Path to save JSON file
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert tuple keys to strings for JSON serialization
        data = {
            "weights": {
                f"{op}:{ctx}": weight
                for (op, ctx), weight in self.weights.items()
            },
            "success_rates": {
                f"{op}:{ctx}": self.get_success_rate(op, ctx)
                for (op, ctx) in self.weights.keys()
            },
            "attempt_counts": {
                f"{op}:{ctx}": count
                for (op, ctx), count in self.attempt_counts.items()
            },
            "statistics": self.get_statistics()
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ K_P saved to {filepath}")

    def load_from_json(self, filepath: str):
        """
        Load learned weights from JSON file.

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Convert string keys back to tuples
        self.weights = {}
        for key_str, weight in data.get("weights", {}).items():
            op, ctx = key_str.split(":", 1)
            self.weights[(op, ctx)] = weight

        # Load attempt counts if available
        self.attempt_counts = {}
        for key_str, count in data.get("attempt_counts", {}).items():
            op, ctx = key_str.split(":", 1)
            self.attempt_counts[(op, ctx)] = count

        # Reconstruct success counts from attempt counts and success rates
        self.success_counts = {}
        for (op, ctx), attempts in self.attempt_counts.items():
            key_str = f"{op}:{ctx}"
            success_rate = data.get("success_rates", {}).get(key_str)
            if success_rate is not None and attempts > 0:
                self.success_counts[(op, ctx)] = int(success_rate * attempts)

        print(f"✅ K_P loaded from {filepath}")
        print(f"   {len(self.weights)} (operation, context) pairs loaded")


if __name__ == "__main__":
    # Test K_P store
    print("=" * 60)
    print("K_P Store Tests")
    print("=" * 60)

    # Import needed modules
    from cig_poc.kr.registry import OPERATIONS
    from cig_poc.kr.context import CONTEXT_TAXONOMY

    # Initialize K_P
    kp = KPStore()
    kp.initialize_uniform(OPERATIONS, CONTEXT_TAXONOMY)

    print(f"\n📊 Initial Statistics:")
    print(f"  Total (op, context) pairs: {len(kp.weights)}")
    print(f"  All weights initialized to: 0.5")

    # Simulate some learning
    print(f"\n🎓 Simulating learning updates...")
    test_updates = [
        ("factors", "small_prime", True),
        ("factors", "small_prime", True),
        ("factors", "small_composite", True),
        ("factors", "large_composite", False),
        ("sum_list", "short_list", True),
        ("sum_list", "short_list", True),
        ("sum_list", "long_list", False),
        ("gcd", "small_numbers", True),
    ]

    for op, ctx, success in test_updates:
        old = kp.get_weight(op, ctx)
        kp.update_weight(op, ctx, success)
        new = kp.get_weight(op, ctx)
        print(f"  {op}:{ctx} {old:.3f} → {new:.3f} ({'✓' if success else '✗'})")

    # Show statistics
    print(f"\n📈 Learning Statistics:")
    stats = kp.get_statistics()
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Pairs with data: {stats['pairs_with_data']}/{stats['total_pairs']}")

    print(f"\n🏆 Top Performers:")
    for (op, ctx), rate, attempts in stats['top_performers'][:5]:
        weight = kp.get_weight(op, ctx)
        print(f"  {op}:{ctx} - {rate*100:.1f}% success ({attempts} attempts), weight={weight:.3f}")

    # Test persistence
    print(f"\n💾 Testing persistence...")
    test_file = "data/results/test_kp.json"
    kp.save_to_json(test_file)

    # Load and verify
    kp2 = KPStore()
    kp2.load_from_json(test_file)
    print(f"\n✅ Verification: Loaded {len(kp2.weights)} pairs")
    print(f"   Sample weight: factors:small_prime = {kp2.get_weight('factors', 'small_prime'):.3f}")
