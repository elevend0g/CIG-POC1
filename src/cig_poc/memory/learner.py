"""
Week 4: Experiential Learner

Learns from episodes without retraining or modifying K_R.
Updates K_P weights based on problem-solving success/failure.
"""

from typing import List, Dict, Any, Optional
import json
import os
from cig_poc.memory.episode import Episode


class ExperientialLearner:
    """
    Learn from episodes without retraining.

    Key Features:
        - Records all problem-solving attempts as episodes
        - Updates K_P weights after each episode
        - Computes learning curves and improvement metrics
        - Persists episodes to disk for analysis

    The learner never modifies K_R (immutable operations),
    only adjusts K_P weights based on experience.
    """

    def __init__(self, K_P, learn_rate: float = 0.1):
        """
        Initialize experiential learner.

        Args:
            K_P: Weight store to update
            learn_rate: Learning rate for weight updates (alpha)
        """
        self.K_P = K_P
        self.learn_rate = learn_rate
        self.episodes: List[Episode] = []

    def record_episode(
        self,
        problem_text: str,
        initial_value: Any,
        path: List[tuple],
        result: Optional[Any],
        expected: Any,
        success: bool,
        nodes_explored: int = 0,
        final_score: float = 0.0
    ) -> Episode:
        """
        Record a problem-solving attempt.

        Args:
            problem_text: Original problem text
            initial_value: Starting value
            path: List of (operation, context) tuples
            result: Final result (or None if no solution)
            expected: Expected answer
            success: Whether goal was satisfied
            nodes_explored: Number of nodes explored in search
            final_score: Final accumulated score

        Returns:
            Created episode
        """
        episode = Episode(
            problem_text=problem_text,
            initial_value=initial_value,
            path=path or [],
            result=result,
            expected=expected,
            goal_satisfied=success,
            nodes_explored=nodes_explored,
            final_score=final_score
        )
        self.episodes.append(episode)
        return episode

    def learn_from_episode(self, episode: Episode):
        """
        Update K_P based on episode outcome.

        For successful episodes:
            - Increase weights for all (operation, context) pairs in path
        For failed episodes:
            - Decrease weights for all (operation, context) pairs attempted

        Args:
            episode: Episode to learn from
        """
        for operation, context in episode.path:
            self.K_P.update_weight(
                operation,
                context,
                episode.goal_satisfied,
                alpha=self.learn_rate
            )

    def learn_from_recent_episodes(self, n: int = 10):
        """
        Update K_P based on recent n episodes.

        Args:
            n: Number of recent episodes to learn from
        """
        recent = self.episodes[-n:] if len(self.episodes) >= n else self.episodes
        for episode in recent:
            self.learn_from_episode(episode)

    def learn_from_all_episodes(self):
        """Batch learning from all recorded episodes"""
        for episode in self.episodes:
            self.learn_from_episode(episode)

    def get_learning_curve(self, window: int = 10) -> List[float]:
        """
        Analyze improvement over episodes using rolling average.

        Args:
            window: Size of rolling window for averaging

        Returns:
            List of rolling average success rates
        """
        if not self.episodes:
            return []

        success_by_episode = [
            1.0 if ep.goal_satisfied else 0.0
            for ep in self.episodes
        ]

        rolling_avg = []
        for i in range(len(success_by_episode)):
            window_start = max(0, i - window + 1)
            window_data = success_by_episode[window_start:i+1]
            rolling_avg.append(sum(window_data) / len(window_data))

        return rolling_avg

    def get_statistics(self) -> Dict:
        """
        Get overall learning statistics.

        Returns:
            Dictionary with metrics:
                - total_episodes: Total episodes recorded
                - success_rate: Overall success rate
                - avg_depth: Average search depth
                - improvement: Improvement from first 10 to last 10 episodes
        """
        if not self.episodes:
            return {
                "total_episodes": 0,
                "success_rate": 0.0,
                "avg_depth": 0.0,
                "improvement": 0.0
            }

        successes = sum(1 for ep in self.episodes if ep.goal_satisfied)
        total = len(self.episodes)
        success_rate = successes / total

        avg_depth = sum(ep.search_depth for ep in self.episodes) / total

        # Calculate improvement (first 10 vs last 10)
        improvement = 0.0
        if total >= 20:
            first_10 = sum(1 for ep in self.episodes[:10] if ep.goal_satisfied) / 10
            last_10 = sum(1 for ep in self.episodes[-10:] if ep.goal_satisfied) / 10
            improvement = last_10 - first_10

        return {
            "total_episodes": total,
            "success_rate": success_rate,
            "avg_depth": avg_depth,
            "avg_nodes_explored": sum(ep.nodes_explored for ep in self.episodes) / total,
            "improvement": improvement,
            "first_10_success": sum(1 for ep in self.episodes[:min(10, total)] if ep.goal_satisfied) / min(10, total),
            "last_10_success": sum(1 for ep in self.episodes[-min(10, total):] if ep.goal_satisfied) / min(10, total)
        }

    def save_episodes(self, filepath: str):
        """
        Persist all episodes to JSON file.

        Args:
            filepath: Path to save episodes
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "total_episodes": len(self.episodes),
            "statistics": self.get_statistics(),
            "episodes": [ep.to_dict() for ep in self.episodes]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Saved {len(self.episodes)} episodes to {filepath}")

    def load_episodes(self, filepath: str):
        """
        Load episodes from JSON file.

        Args:
            filepath: Path to episodes file
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.episodes = [
            Episode.from_dict(ep_data)
            for ep_data in data["episodes"]
        ]

        print(f"✅ Loaded {len(self.episodes)} episodes from {filepath}")

    def get_failed_episodes(self) -> List[Episode]:
        """Get all episodes where goal was not satisfied"""
        return [ep for ep in self.episodes if not ep.goal_satisfied]

    def get_successful_episodes(self) -> List[Episode]:
        """Get all episodes where goal was satisfied"""
        return [ep for ep in self.episodes if ep.goal_satisfied]

    def print_summary(self):
        """Print human-readable summary of learning progress"""
        stats = self.get_statistics()

        print("=" * 60)
        print("Experiential Learning Summary")
        print("=" * 60)
        print(f"\n📊 Overall Statistics:")
        print(f"  Total episodes: {stats['total_episodes']}")
        print(f"  Success rate: {stats['success_rate']*100:.1f}%")
        print(f"  Avg search depth: {stats['avg_depth']:.2f}")
        print(f"  Avg nodes explored: {stats['avg_nodes_explored']:.1f}")

        if stats['total_episodes'] >= 20:
            print(f"\n📈 Learning Progress:")
            print(f"  First 10 episodes: {stats['first_10_success']*100:.1f}% success")
            print(f"  Last 10 episodes: {stats['last_10_success']*100:.1f}% success")
            print(f"  Improvement: {stats['improvement']*100:+.1f}%")

        # Show recent failures for debugging
        failed = self.get_failed_episodes()
        if failed:
            print(f"\n❌ Recent Failures ({len(failed)} total):")
            for ep in failed[-3:]:
                print(f"  - {ep.problem_text[:60]}")
                if ep.path:
                    print(f"    Attempted: {[op for op, ctx in ep.path]}")
                print(f"    Expected: {ep.expected}, Got: {ep.result}")


if __name__ == "__main__":
    # Test experiential learner
    print("=" * 60)
    print("Experiential Learner Tests")
    print("=" * 60)

    # Import required modules
    import sys
    sys.path.insert(0, "/home/jay/ag0/cig/src")
    from cig_poc.kp.store import KPStore
    from cig_poc.kr.registry import OPERATIONS
    from cig_poc.kr.context import CONTEXT_TAXONOMY

    # Initialize K_P
    kp = KPStore()
    kp.initialize_uniform(OPERATIONS, CONTEXT_TAXONOMY)

    # Create learner
    learner = ExperientialLearner(kp, learn_rate=0.1)

    # Simulate some episodes
    print("\n🎓 Simulating 5 problem-solving episodes...")

    episodes_data = [
        ("What is the sum of all factors of 12?", 12, [("factors", "small_composite"), ("sum_list", "short_list")], 28, 28, True),
        ("How many factors does 24 have?", 24, [("factors", "small_composite"), ("count", "short_list")], 8, 8, True),
        ("Is 17 prime?", 17, [("is_prime", "small_number")], True, True, True),
        ("What is the GCD of 12 and 18?", (12, 18), [("gcd", "small_numbers")], 6, 6, True),
        ("What is the sum of all factors of 100?", 100, [("factors", "large_composite")], 150, 217, False),  # Failed
    ]

    for problem_text, initial, path, result, expected, success in episodes_data:
        episode = learner.record_episode(
            problem_text=problem_text,
            initial_value=initial,
            path=path,
            result=result,
            expected=expected,
            success=success,
            nodes_explored=3,
            final_score=0.5
        )
        print(f"  {episode}")

        # Learn from this episode
        learner.learn_from_episode(episode)

    # Show statistics
    print("\n" + "=" * 60)
    learner.print_summary()

    # Show learning curve
    print("\n📈 Learning Curve (rolling average, window=3):")
    curve = learner.get_learning_curve(window=3)
    for i, avg in enumerate(curve):
        print(f"  Episode {i+1}: {avg*100:.1f}%")

    # Test persistence
    print("\n💾 Testing persistence...")
    test_file = "data/results/test_episodes.json"
    learner.save_episodes(test_file)

    # Load and verify
    learner2 = ExperientialLearner(kp)
    learner2.load_episodes(test_file)
    print(f"\n✅ Verification: Loaded {len(learner2.episodes)} episodes")
    print(f"   Statistics match: {learner.get_statistics() == learner2.get_statistics()}")
