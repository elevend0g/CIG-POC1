#!/usr/bin/env python3
"""
Week 4: Training Script - Integrated Learning Loop

Runs beam search solver on training problems, records episodes,
and learns from experience by updating K_P weights.
"""

import sys
sys.path.insert(0, "/home/jay/ag0/cig/src")

import json
import time
from cig_poc.kr.registry import OPERATIONS
from cig_poc.kr.context import CONTEXT_TAXONOMY
from cig_poc.kp.store import KPStore
from cig_poc.search.beam import BeamSearchSolver
from cig_poc.memory.learner import ExperientialLearner


def load_problems(filepath: str) -> list:
    """Load problems from JSONL file"""
    problems = []
    with open(filepath, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def run_training(
    training_file: str,
    num_problems: int = 100,
    learn_rate: float = 0.1,
    checkpoint_every: int = 20,
    output_dir: str = "data/results"
):
    """
    Run training loop on problems.

    Args:
        training_file: Path to training problems JSONL
        num_problems: Number of problems to train on
        learn_rate: Learning rate for K_P updates
        checkpoint_every: Save checkpoint every N problems
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("CIG Proof of Concept - Training Loop")
    print("=" * 70)

    # Load problems
    print(f"\n[1/5] Loading training problems from {training_file}...")
    problems = load_problems(training_file)
    print(f"✅ Loaded {len(problems)} problems")
    print(f"   Using first {num_problems} problems")
    problems = problems[:num_problems]

    # Initialize K_R (immutable operations)
    print(f"\n[2/5] Initializing K_R (immutable operations)...")
    print(f"✅ K_R loaded: {len(OPERATIONS)} operations")

    # Initialize K_P (learnable weights)
    print(f"\n[3/5] Initializing K_P (learnable weights)...")
    kp = KPStore()
    kp.initialize_uniform(OPERATIONS, CONTEXT_TAXONOMY)

    # Initialize solver and learner
    print(f"\n[4/5] Initializing solver and learner...")
    solver = BeamSearchSolver(OPERATIONS, kp, beam_width=10, max_depth=5)
    learner = ExperientialLearner(kp, learn_rate=learn_rate)
    print(f"✅ Beam search solver ready (width=10, max_depth=5)")
    print(f"✅ Experiential learner ready (α={learn_rate})")

    # Training loop
    print(f"\n[5/5] Starting training loop...")
    print("=" * 70)

    start_time = time.time()
    checkpoint_times = []

    for i, problem in enumerate(problems):
        problem_num = i + 1

        # Extract problem details
        problem_text = problem["text"]
        expected = problem["expected"]

        # Extract initial value from problem
        # For most problems, this is the first number in params
        if "n" in problem["params"]:
            initial_value = problem["params"]["n"]
        elif "a" in problem["params"]:
            initial_value = (problem["params"]["a"], problem["params"]["b"])
        else:
            initial_value = problem["params"][list(problem["params"].keys())[0]]

        # Define goal predicate
        def goal(result):
            return result == expected

        # Solve problem
        try:
            path, score, trace = solver.solve(
                problem_text=problem_text,
                initial_value=initial_value,
                goal=goal
            )

            # Check if solution is correct
            if trace and trace["result"] == expected:
                success = True
                result = trace["result"]
                nodes = solver.search_stats["nodes_explored"]
            else:
                success = False
                result = trace["result"] if trace else None
                nodes = solver.search_stats.get("nodes_explored", 0)

        except Exception as e:
            # Solver failed
            success = False
            result = None
            path = []
            score = 0.0
            nodes = 0
            print(f"\n⚠️  Problem {problem_num} crashed: {str(e)[:50]}")

        # Record episode
        episode = learner.record_episode(
            problem_text=problem_text,
            initial_value=initial_value,
            path=path or [],
            result=result,
            expected=expected,
            success=success,
            nodes_explored=nodes,
            final_score=score
        )

        # Learn from episode
        learner.learn_from_episode(episode)

        # Print progress
        status = "✓" if success else "✗"
        print(f"{problem_num:3d}/100 {status} {problem_text[:60]:<60} "
              f"depth={episode.search_depth} nodes={nodes}")

        # Checkpoint
        if problem_num % checkpoint_every == 0:
            elapsed = time.time() - start_time
            checkpoint_times.append(elapsed)

            stats = learner.get_statistics()
            print(f"\n{'='*70}")
            print(f"Checkpoint at {problem_num} problems:")
            print(f"  Success rate: {stats['success_rate']*100:.1f}%")
            print(f"  Avg depth: {stats['avg_depth']:.2f}")
            print(f"  Elapsed time: {elapsed:.1f}s")
            print(f"{'='*70}\n")

            # Save checkpoint
            kp.save_to_json(f"{output_dir}/kp_checkpoint_{problem_num}.json")

    # Final statistics
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    stats = learner.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"  Total problems: {stats['total_episodes']}")
    print(f"  Success rate: {stats['success_rate']*100:.1f}%")
    print(f"  Avg search depth: {stats['avg_depth']:.2f}")
    print(f"  Avg nodes explored: {stats['avg_nodes_explored']:.1f}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Time per problem: {total_time/len(problems):.2f}s")

    if stats['total_episodes'] >= 20:
        print(f"\n📈 Learning Progress:")
        print(f"  First 10 problems: {stats['first_10_success']*100:.1f}% success")
        print(f"  Last 10 problems: {stats['last_10_success']*100:.1f}% success")
        print(f"  Improvement: {stats['improvement']*100:+.1f}%")

    # Save results
    print(f"\n💾 Saving results...")
    kp.save_to_json(f"{output_dir}/kp_final.json")
    learner.save_episodes(f"{output_dir}/episodes_all.json")

    # Save learning curve
    curve = learner.get_learning_curve(window=10)
    with open(f"{output_dir}/learning_curve.json", "w") as f:
        json.dump({
            "curve": curve,
            "statistics": stats,
            "checkpoint_times": checkpoint_times
        }, f, indent=2)

    print(f"✅ Results saved to {output_dir}/")
    print(f"   - kp_final.json (final learned weights)")
    print(f"   - episodes_all.json (all problem attempts)")
    print(f"   - learning_curve.json (learning progress)")

    # Show failed problems for debugging
    failed = learner.get_failed_episodes()
    if failed and len(failed) <= 20:
        print(f"\n❌ Failed Problems ({len(failed)} total):")
        for ep in failed[:10]:
            print(f"   {ep.problem_text[:65]}")
            if ep.path:
                print(f"      Attempted: {[op for op, ctx in ep.path]}")
            print(f"      Expected: {ep.expected}, Got: {ep.result}")

    return learner, kp, stats


if __name__ == "__main__":
    # Run training on first 100 problems
    learner, kp, stats = run_training(
        training_file="data/training/arithmetic_100.jsonl",
        num_problems=100,
        learn_rate=0.1,
        checkpoint_every=20,
        output_dir="data/results"
    )

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 Week 4 Complete!")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  ✓ Trained on 100 arithmetic problems")
    print(f"  ✓ Final accuracy: {stats['success_rate']*100:.1f}%")
    print(f"  ✓ K_P learned from {stats['total_episodes']} episodes")
    print(f"  ✓ No modifications to K_R (immutable operations)")
    print(f"\nNext: Week 5 - Evaluation & Baseline Comparison")
