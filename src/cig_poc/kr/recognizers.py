"""
Week 1 Part B: Pattern Recognition Layer

Keyword-based pattern recognition to map problem text to candidate operations.
This reduces search space from ~30 operations to ~3-5 candidates per problem.
"""

from typing import List, Tuple, Dict
import re

# Pattern rules: keyword → operation mapping with confidence scores
PATTERN_RULES = {
    # Generators
    "factors": {
        "keywords": ["factor", "divisor", "divide", "divides"],
        "confidence": 0.95,
        "category": "generator"
    },
    "prime_factors": {
        "keywords": ["prime factor", "factorization", "prime divisor"],
        "confidence": 0.95,
        "category": "generator"
    },
    "range_to_n": {
        "keywords": ["from 1 to", "up to", "range"],
        "confidence": 0.85,
        "category": "generator"
    },

    # Aggregators
    "sum_list": {
        "keywords": ["sum", "total", "add all", "add together"],
        "confidence": 0.90,
        "category": "aggregator"
    },
    "product_list": {
        "keywords": ["product", "multiply all", "multiplication"],
        "confidence": 0.90,
        "category": "aggregator"
    },
    "count": {
        "keywords": ["how many", "count", "number of"],
        "confidence": 0.95,
        "category": "aggregator"
    },
    "max_list": {
        "keywords": ["largest", "maximum", "greatest", "biggest"],
        "confidence": 0.85,
        "category": "aggregator"
    },
    "max_excluding_self": {
        "keywords": ["other than", "(other than"],
        "confidence": 0.95,
        "category": "aggregator"
    },
    "min_list": {
        "keywords": ["smallest", "minimum", "least"],
        "confidence": 0.85,
        "category": "aggregator"
    },
    "average": {
        "keywords": ["average", "mean"],
        "confidence": 0.95,
        "category": "aggregator"
    },

    # Number Theory
    "gcd": {
        "keywords": ["gcd", "greatest common divisor", "common divisor"],
        "confidence": 0.95,
        "category": "number_theory"
    },
    "lcm": {
        "keywords": ["lcm", "least common multiple", "common multiple"],
        "confidence": 0.95,
        "category": "number_theory"
    },
    "is_prime": {
        "keywords": ["prime", "is prime"],
        "confidence": 0.95,
        "category": "predicate"
    },

    # Predicates
    "is_even": {
        "keywords": ["even"],
        "confidence": 0.90,
        "category": "predicate"
    },
    "is_odd": {
        "keywords": ["odd"],
        "confidence": 0.90,
        "category": "predicate"
    },

    # Filters
    "filter_even": {
        "keywords": ["even factor", "even divisor", "even number"],
        "confidence": 0.90,
        "category": "filter"
    },
    "filter_odd": {
        "keywords": ["odd factor", "odd divisor", "odd number"],
        "confidence": 0.90,
        "category": "filter"
    },
    "filter_primes": {
        "keywords": ["prime factor", "prime number"],
        "confidence": 0.85,
        "category": "filter"
    },

    # Arithmetic
    "add": {
        "keywords": ["add", "plus", "sum of {n} and {m}"],
        "confidence": 0.85,
        "category": "arithmetic"
    },
    "multiply": {
        "keywords": ["multiply", "times", "product of {n} and {m}"],
        "confidence": 0.85,
        "category": "arithmetic"
    },
    "subtract": {
        "keywords": ["subtract", "minus", "difference"],
        "confidence": 0.85,
        "category": "arithmetic"
    },
    "divide": {
        "keywords": ["divide", "quotient"],
        "confidence": 0.85,
        "category": "arithmetic"
    },

    # Transforms
    "square": {
        "keywords": ["square", "squared"],
        "confidence": 0.90,
        "category": "transform"
    },
    "double": {
        "keywords": ["double", "twice"],
        "confidence": 0.85,
        "category": "transform"
    },
}


def simple_pattern_recognizer(text: str) -> List[Tuple[str, float]]:
    """
    Keyword-based pattern recognition.
    Maps problem text to candidate operations with confidence scores.

    Args:
        text: Problem text to analyze

    Returns:
        List of (operation_name, confidence) tuples sorted by confidence (descending)

    Example:
        >>> recognize("What is the sum of all factors of 12?")
        [("factors", 0.95), ("sum_list", 0.90)]
    """
    text_lower = text.lower()
    candidates = []

    # Sort operations by maximum keyword length (longest first) to prioritize specific patterns
    sorted_ops = sorted(
        PATTERN_RULES.items(),
        key=lambda x: max(len(kw) for kw in x[1]["keywords"]),
        reverse=True
    )

    for op_name, rules in sorted_ops:
        for keyword in rules["keywords"]:
            if keyword in text_lower:
                candidates.append((op_name, rules["confidence"]))
                break  # Match this operation once per text

    # Sort by confidence (descending)
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def extract_numbers(text: str) -> List[int]:
    """
    Extract all numbers from problem text.

    Args:
        text: Problem text

    Returns:
        List of integers found in text

    Example:
        >>> extract_numbers("What is the GCD of 12 and 18?")
        [12, 18]
    """
    # Find all numbers in text
    numbers = re.findall(r'\b\d+\b', text)
    return [int(n) for n in numbers]


def recognize_with_numbers(text: str) -> Dict:
    """
    Full pattern recognition including operations and numbers.

    Args:
        text: Problem text

    Returns:
        Dictionary with:
            - operations: List of (operation_name, confidence) tuples
            - numbers: List of integers extracted from text
            - text: Original problem text

    Example:
        >>> recognize_with_numbers("What is the sum of all factors of 12?")
        {
            'operations': [('factors', 0.95), ('sum_list', 0.90)],
            'numbers': [12],
            'text': 'What is the sum of all factors of 12?'
        }
    """
    operations = simple_pattern_recognizer(text)
    numbers = extract_numbers(text)

    return {
        "operations": operations,
        "numbers": numbers,
        "text": text
    }


def analyze_problem_set(problems: List[Dict]) -> Dict:
    """
    Analyze a problem set to compute pattern recognition statistics.

    Args:
        problems: List of problem dictionaries (from JSONL)

    Returns:
        Statistics about pattern recognition coverage
    """
    total = len(problems)
    recognized = 0
    coverage = {}

    for problem in problems:
        result = recognize_with_numbers(problem["text"])
        ops_found = len(result["operations"])

        if ops_found > 0:
            recognized += 1

        for op_name, _ in result["operations"]:
            coverage[op_name] = coverage.get(op_name, 0) + 1

    return {
        "total_problems": total,
        "problems_with_matches": recognized,
        "coverage_rate": recognized / total if total > 0 else 0,
        "operation_usage": coverage
    }


if __name__ == "__main__":
    # Test pattern recognition
    test_cases = [
        "What is the sum of all factors of 12?",
        "What is the GCD of 12 and 18?",
        "Is 17 prime?",
        "How many factors does 24 have?",
        "What is the product of all factors of 6?",
        "What is the largest factor of 20 (other than 20 itself)?",
        "What is the sum of even factors of 30?",
        "What is the LCM of 15 and 25?",
    ]

    print("=" * 60)
    print("Pattern Recognition Tests")
    print("=" * 60)

    for text in test_cases:
        result = recognize_with_numbers(text)
        print(f"\nProblem: {text}")
        print(f"Numbers: {result['numbers']}")
        print(f"Operations: {result['operations']}")

    # Test with real problem set
    print("\n" + "=" * 60)
    print("Problem Set Analysis")
    print("=" * 60)

    import json
    import os

    training_file = "data/training/arithmetic_100.jsonl"
    if os.path.exists(training_file):
        problems = []
        with open(training_file, 'r') as f:
            for line in f:
                problems.append(json.loads(line))

        stats = analyze_problem_set(problems)
        print(f"\nTotal problems: {stats['total_problems']}")
        print(f"Problems with matches: {stats['problems_with_matches']}")
        print(f"Coverage rate: {stats['coverage_rate']*100:.1f}%")
        print(f"\nOperation usage:")
        for op, count in sorted(stats['operation_usage'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {op}: {count} problems")
    else:
        print(f"⚠️  Training file not found: {training_file}")
