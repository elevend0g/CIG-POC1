"""
Week 4: Episode Recording

Dataclass for recording problem-solving attempts.
Enables episodic learning without modifying K_R.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional
import time


@dataclass
class Episode:
    """
    Record of a single problem-solving attempt.

    Stores complete trace of problem-solving process including:
    - Problem description
    - Operations attempted
    - Contexts encountered
    - Final result
    - Success/failure

    This enables the system to learn from experience without
    modifying the immutable K_R operations.
    """

    problem_text: str
    initial_value: Any
    path: List[Tuple[str, str]]  # [(operation, context), ...]
    result: Optional[Any]
    expected: Any
    goal_satisfied: bool
    timestamp: float = field(default_factory=time.time)
    search_depth: int = 0
    nodes_explored: int = 0
    final_score: float = 0.0

    def __post_init__(self):
        """Calculate derived fields"""
        self.search_depth = len(self.path)

    def to_dict(self) -> dict:
        """Convert episode to dictionary for JSON serialization"""
        return {
            "problem_text": self.problem_text,
            "initial_value": self.initial_value,
            "path": [(op, ctx) for op, ctx in self.path],
            "result": self.result,
            "expected": self.expected,
            "goal_satisfied": self.goal_satisfied,
            "timestamp": self.timestamp,
            "search_depth": self.search_depth,
            "nodes_explored": self.nodes_explored,
            "final_score": self.final_score
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Create episode from dictionary"""
        return cls(
            problem_text=data["problem_text"],
            initial_value=data["initial_value"],
            path=[(op, ctx) for op, ctx in data["path"]],
            result=data["result"],
            expected=data["expected"],
            goal_satisfied=data["goal_satisfied"],
            timestamp=data["timestamp"],
            search_depth=data["search_depth"],
            nodes_explored=data["nodes_explored"],
            final_score=data["final_score"]
        )

    def __repr__(self) -> str:
        """Human-readable representation"""
        status = "✓" if self.goal_satisfied else "✗"
        return (
            f"Episode({status} {self.problem_text[:50]}... "
            f"depth={self.search_depth}, result={self.result})"
        )


if __name__ == "__main__":
    # Test episode creation
    print("=" * 60)
    print("Episode Recording Tests")
    print("=" * 60)

    # Create a test episode
    episode = Episode(
        problem_text="What is the sum of all factors of 12?",
        initial_value=12,
        path=[("factors", "small_composite"), ("sum_list", "short_list")],
        result=28,
        expected=28,
        goal_satisfied=True,
        nodes_explored=3,
        final_score=0.0312
    )

    print(f"\n📝 Episode created:")
    print(f"  {episode}")
    print(f"\n  Details:")
    print(f"    Problem: {episode.problem_text}")
    print(f"    Initial value: {episode.initial_value}")
    print(f"    Path: {episode.path}")
    print(f"    Result: {episode.result}")
    print(f"    Expected: {episode.expected}")
    print(f"    Success: {episode.goal_satisfied}")
    print(f"    Depth: {episode.search_depth}")
    print(f"    Nodes explored: {episode.nodes_explored}")

    # Test serialization
    print(f"\n💾 Testing serialization:")
    data = episode.to_dict()
    print(f"  Converted to dict: {len(data)} fields")

    episode2 = Episode.from_dict(data)
    print(f"  Restored from dict: {episode2}")
    print(f"  Match: {episode.problem_text == episode2.problem_text}")
