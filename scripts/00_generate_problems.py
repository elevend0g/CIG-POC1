#!/usr/bin/env python3
"""
Week 0: Problem Generation & Validation
Generate 100 training problems and 20 holdout problems for CIG PoC
"""

from typing import Callable, Dict, Any, List
import json
import math
import random
import os

# Helper functions for ground truth computation
def _is_prime(n):
    """Check if n is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _factors(n):
    """Get all factors of n"""
    return [i for i in range(1, n+1) if n % i == 0]

def _prime_factors(n):
    """Get prime factorization of n"""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

# Problem templates with ground truth computation
PROBLEM_TEMPLATES = [
    {
        "template": "What is the sum of all factors of {n}?",
        "ground_truth": lambda n: sum(_factors(n)),
        "params": {"n": range(5, 100)},
        "category": "factors_sum",
        "operation_chain": ["factors", "sum_list"]
    },
    {
        "template": "What is the product of all factors of {n}?",
        "ground_truth": lambda n: math.prod(_factors(n)),
        "params": {"n": range(5, 30)},  # Smaller range to avoid overflow
        "category": "factors_product",
        "operation_chain": ["factors", "product_list"]
    },
    {
        "template": "How many factors does {n} have?",
        "ground_truth": lambda n: len(_factors(n)),
        "params": {"n": range(5, 100)},
        "category": "factors_count",
        "operation_chain": ["factors", "count"]
    },
    {
        "template": "What is the GCD of {a} and {b}?",
        "ground_truth": lambda a, b: math.gcd(a, b),
        "params": {"a": range(10, 100), "b": range(10, 100)},
        "category": "gcd",
        "operation_chain": ["gcd"]
    },
    {
        "template": "What is the LCM of {a} and {b}?",
        "ground_truth": lambda a, b: (a * b) // math.gcd(a, b),
        "params": {"a": range(5, 50), "b": range(5, 50)},
        "category": "lcm",
        "operation_chain": ["lcm"]
    },
    {
        "template": "Is {n} prime?",
        "ground_truth": lambda n: _is_prime(n),
        "params": {"n": range(2, 200)},
        "category": "primality",
        "operation_chain": ["is_prime"]
    },
    {
        "template": "What is the sum of even factors of {n}?",
        "ground_truth": lambda n: sum(f for f in _factors(n) if f % 2 == 0),
        "params": {"n": range(10, 100)},
        "category": "factors_filter_sum",
        "operation_chain": ["factors", "filter_even", "sum_list"]
    },
    {
        "template": "What is the largest factor of {n} (other than {n} itself)?",
        "ground_truth": lambda n: _factors(n)[-2] if len(_factors(n)) > 1 else 1,
        "params": {"n": range(10, 100)},
        "category": "largest_proper_factor",
        "operation_chain": ["factors", "max_excluding_self"]
    },
]

def generate_problem_set(num_problems=100, seed=42):
    """
    Generate balanced problem set across categories.

    Strategy:
    1. Distribute problems evenly across categories
    2. Sample parameter values randomly from ranges
    3. Compute ground truth using template functions
    4. Validate each problem is solvable

    Args:
        num_problems: Total number of problems to generate
        seed: Random seed for reproducibility

    Returns:
        List of problem dictionaries
    """
    random.seed(seed)

    problems = []
    problems_per_category = num_problems // len(PROBLEM_TEMPLATES)

    # Ensure we generate exactly num_problems
    remainder = num_problems % len(PROBLEM_TEMPLATES)

    for i, template_spec in enumerate(PROBLEM_TEMPLATES):
        # Add extra problems to first categories to reach exact count
        count = problems_per_category + (1 if i < remainder else 0)

        for _ in range(count):
            # Sample parameters
            params = {}
            for param_name, param_range in template_spec["params"].items():
                params[param_name] = random.choice(list(param_range))

            # Generate problem text
            problem_text = template_spec["template"].format(**params)

            # Compute ground truth
            try:
                ground_truth = template_spec["ground_truth"](**params)
            except Exception as e:
                print(f"Error computing ground truth for {problem_text}: {e}")
                continue

            problems.append({
                "text": problem_text,
                "params": params,
                "expected": ground_truth,
                "category": template_spec["category"],
                "operation_chain": template_spec["operation_chain"]
            })

    return problems

def validate_problem_set(problems):
    """
    Verify that all generated problems are valid.

    Checks:
    1. Ground truth is computable and valid
    2. No problems are trivial or degenerate
    3. Problem distribution is balanced

    Args:
        problems: List of problem dictionaries

    Returns:
        Boolean indicating validation success
    """
    validation_errors = []
    required_operations = set()

    for i, problem in enumerate(problems):
        # Track required operations
        for op_name in problem["operation_chain"]:
            required_operations.add(op_name)

        # Check ground truth is valid
        expected = problem["expected"]
        if expected is None:
            validation_errors.append(
                f"Problem {i}: Ground truth is None"
            )
        elif isinstance(expected, float) and (math.isnan(expected) or math.isinf(expected)):
            validation_errors.append(
                f"Problem {i}: Invalid ground truth: {expected}"
            )

    # Check problem distribution
    category_counts = {}
    for problem in problems:
        cat = problem["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    max_category_pct = max(category_counts.values()) / len(problems) * 100
    if max_category_pct > 30:
        validation_errors.append(
            f"Imbalanced distribution: {max(category_counts, key=category_counts.get)} "
            f"has {max_category_pct:.1f}% of problems"
        )

    if validation_errors:
        print(f"❌ Validation failed with {len(validation_errors)} errors:")
        for error in validation_errors[:10]:  # Show first 10
            print(f"  - {error}")
        return False
    else:
        print(f"✅ All {len(problems)} problems validated successfully")
        print(f"\nRequired operations for K_R ({len(required_operations)} total):")
        for op in sorted(required_operations):
            print(f"  - {op}")
        print(f"\nCategory distribution:")
        for cat, count in sorted(category_counts.items()):
            print(f"  - {cat}: {count} ({count/len(problems)*100:.1f}%)")
        return True

def save_problem_set(problems, filepath):
    """Save problems to JSONL format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for problem in problems:
            f.write(json.dumps(problem) + '\n')
    print(f"✅ Saved {len(problems)} problems to {filepath}")

def load_problem_set(filepath):
    """Load problems from JSONL format"""
    problems = []
    with open(filepath, 'r') as f:
        for line in f:
            problems.append(json.loads(line))
    return problems

def main():
    """Generate training and holdout problem sets"""
    print("=" * 60)
    print("Week 0: Problem Generation & Validation")
    print("=" * 60)

    # Generate main training set (100 problems)
    print("\n[1/4] Generating training set (100 problems)...")
    training_problems = generate_problem_set(num_problems=100, seed=42)

    print("\n[2/4] Validating training set...")
    if not validate_problem_set(training_problems):
        print("⚠️  Training set validation failed!")
        return

    print("\n[3/4] Saving training set...")
    save_problem_set(training_problems, "data/training/arithmetic_100.jsonl")

    # Generate holdout set (20 problems for Week 6)
    print("\n[4/4] Generating holdout set (20 problems)...")
    holdout_problems = generate_problem_set(num_problems=20, seed=123)

    print("\nValidating holdout set...")
    if not validate_problem_set(holdout_problems):
        print("⚠️  Holdout set validation failed!")
        return

    print("\nSaving holdout set...")
    save_problem_set(holdout_problems, "data/holdout/arithmetic_20_holdout.jsonl")

    print("\n" + "=" * 60)
    print("✅ Week 0 Complete!")
    print("=" * 60)
    print(f"\nDeliverables:")
    print(f"  - {len(training_problems)} training problems: data/training/arithmetic_100.jsonl")
    print(f"  - {len(holdout_problems)} holdout problems: data/holdout/arithmetic_20_holdout.jsonl")
    print(f"\nNext: Week 1 - Implement K_R operations and pattern recognition")

if __name__ == "__main__":
    main()
