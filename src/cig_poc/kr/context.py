"""
Week 1 Part C: Context Extraction Rules

Determines context keys for K_P lookup: Dict[(operation, context), weight]
This enables learned weights to differentiate between usage patterns.
"""

from typing import Any, Union, List


def _is_prime(n: int) -> bool:
    """Helper: Check if n is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def extract_context(value: Any, operation: str) -> str:
    """
    Determine context key for K_P lookup.

    K_P is organized as: Dict[(operation, context), weight]

    This function determines which context a value falls into,
    enabling learned weights to differentiate between usage patterns.

    Args:
        value: The input being operated on (number, list, tuple, etc.)
        operation: The name of the operation being applied

    Returns:
        context_key: String used to look up K_P[(operation, context_key)]

    Example:
        >>> extract_context(17, "factors")
        "small_prime"  # Factors operation on small prime number

        >>> extract_context([1, 2, 3], "sum_list")
        "short_list"  # Sum operation on short list
    """

    # ========== Generators ==========
    if operation == "factors":
        if not isinstance(value, int):
            return "invalid"
        if value < 100:
            if _is_prime(value):
                return "small_prime"  # Factors are fast: [1, p]
            else:
                return "small_composite"  # Factors vary
        else:
            if _is_prime(value):
                return "large_prime"  # Factors fast but computation slower
            else:
                return "large_composite"

    elif operation == "prime_factors":
        if not isinstance(value, int):
            return "invalid"
        if value < 100:
            return "small_number"
        elif value < 10000:
            return "medium_number"
        else:
            return "large_number"

    elif operation == "range_to_n":
        if not isinstance(value, int):
            return "invalid"
        if value < 50:
            return "small_range"
        elif value < 500:
            return "medium_range"
        else:
            return "large_range"

    # ========== Aggregators ==========
    elif operation in ["sum_list", "product_list", "count", "average"]:
        if not isinstance(value, list):
            return "invalid"
        if len(value) < 10:
            return "short_list"
        elif len(value) < 50:
            return "medium_list"
        else:
            return "long_list"

    elif operation in ["min_list", "max_list", "max_excluding_self", "first", "last"]:
        if not isinstance(value, list):
            return "invalid"
        if len(value) < 10:
            return "short_list"
        else:
            return "long_list"

    # ========== Number Theory ==========
    elif operation == "gcd":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return "invalid"
        a, b = value
        if a < 100 and b < 100:
            return "small_numbers"
        elif a < 1000 or b < 1000:
            return "medium_numbers"
        else:
            return "large_numbers"

    elif operation == "lcm":
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return "invalid"
        a, b = value
        if a < 100 and b < 100:
            return "small_numbers"
        elif a < 1000 or b < 1000:
            return "medium_numbers"
        else:
            return "large_numbers"

    elif operation == "is_prime":
        if not isinstance(value, int):
            return "invalid"
        if value < 100:
            return "small_number"
        elif value < 10000:
            return "medium_number"
        else:
            return "large_number"

    # ========== Predicates ==========
    elif operation in ["is_even", "is_odd"]:
        if not isinstance(value, int):
            return "invalid"
        return "number"

    elif operation in ["greater_than", "less_than", "equals"]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return "invalid"
        return "comparison"

    # ========== Filters ==========
    elif operation in ["filter_even", "filter_odd", "filter_primes"]:
        if not isinstance(value, list):
            return "invalid"
        if len(value) < 10:
            return "short_list"
        elif len(value) < 50:
            return "medium_list"
        else:
            return "long_list"

    # ========== Transforms ==========
    elif operation in ["square", "double", "half", "increment", "decrement"]:
        if not isinstance(value, int):
            return "invalid"
        if value < 100:
            return "small_number"
        elif value < 1000:
            return "medium_number"
        else:
            return "large_number"

    # ========== Arithmetic ==========
    elif operation in ["add", "subtract", "multiply", "divide", "modulo"]:
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            return "invalid"
        a, b = value
        if a < 100 and b < 100:
            return "small_numbers"
        elif a < 1000 or b < 1000:
            return "medium_numbers"
        else:
            return "large_numbers"

    # Default fallback
    return "generic"


# Context taxonomy (for documentation and validation)
CONTEXT_TAXONOMY = {
    # Generators
    "factors": ["small_prime", "small_composite", "large_prime", "large_composite", "invalid"],
    "prime_factors": ["small_number", "medium_number", "large_number", "invalid"],
    "range_to_n": ["small_range", "medium_range", "large_range", "invalid"],

    # Aggregators
    "sum_list": ["short_list", "medium_list", "long_list", "invalid"],
    "product_list": ["short_list", "medium_list", "long_list", "invalid"],
    "count": ["short_list", "medium_list", "long_list", "invalid"],
    "average": ["short_list", "medium_list", "long_list", "invalid"],
    "min_list": ["short_list", "long_list", "invalid"],
    "max_list": ["short_list", "long_list", "invalid"],
    "max_excluding_self": ["short_list", "long_list", "invalid"],
    "first": ["short_list", "long_list", "invalid"],
    "last": ["short_list", "long_list", "invalid"],

    # Number Theory
    "gcd": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "lcm": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "is_prime": ["small_number", "medium_number", "large_number", "invalid"],

    # Predicates
    "is_even": ["number", "invalid"],
    "is_odd": ["number", "invalid"],
    "greater_than": ["comparison", "invalid"],
    "less_than": ["comparison", "invalid"],
    "equals": ["comparison", "invalid"],

    # Filters
    "filter_even": ["short_list", "medium_list", "long_list", "invalid"],
    "filter_odd": ["short_list", "medium_list", "long_list", "invalid"],
    "filter_primes": ["short_list", "medium_list", "long_list", "invalid"],

    # Transforms
    "square": ["small_number", "medium_number", "large_number", "invalid"],
    "double": ["small_number", "medium_number", "large_number", "invalid"],
    "half": ["small_number", "medium_number", "large_number", "invalid"],
    "increment": ["small_number", "medium_number", "large_number", "invalid"],
    "decrement": ["small_number", "medium_number", "large_number", "invalid"],

    # Arithmetic
    "add": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "subtract": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "multiply": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "divide": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "modulo": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
}


def get_all_contexts_for_operation(operation: str) -> List[str]:
    """
    Get all possible context keys for a given operation.

    Args:
        operation: Operation name

    Returns:
        List of context keys (excluding "invalid")
    """
    if operation in CONTEXT_TAXONOMY:
        return [ctx for ctx in CONTEXT_TAXONOMY[operation] if ctx != "invalid"]
    return ["generic"]


def validate_context_extraction():
    """Validate that context extraction is working correctly"""
    test_cases = [
        (17, "factors", "small_prime"),
        (24, "factors", "small_composite"),
        ([1, 2, 3], "sum_list", "short_list"),
        ([1]*50, "sum_list", "long_list"),
        ((12, 18), "gcd", "small_numbers"),
        (7, "is_prime", "small_number"),
        (5000, "is_prime", "medium_number"),
        (15000, "is_prime", "large_number"),
    ]

    errors = []
    for value, operation, expected_context in test_cases:
        result = extract_context(value, operation)
        if result != expected_context:
            errors.append(
                f"extract_context({value}, '{operation}') = '{result}', expected '{expected_context}'"
            )

    if errors:
        print("❌ Context extraction validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ Context extraction validation passed")
        return True


if __name__ == "__main__":
    print("=" * 60)
    print("Context Extraction Tests")
    print("=" * 60)

    # Run validation
    validate_context_extraction()

    # Test examples
    print("\n📋 Example Context Extractions:")
    test_examples = [
        (12, "factors"),
        (17, "factors"),
        ([1, 2, 3, 4, 5], "sum_list"),
        ((12, 18), "gcd"),
        (7, "is_prime"),
        (5000, "is_prime"),
        ([1]*15, "product_list"),
    ]

    for value, operation in test_examples:
        context = extract_context(value, operation)
        print(f"  extract_context({value!r}, '{operation}') = '{context}'")

    # Show taxonomy
    print("\n📊 Context Taxonomy Coverage:")
    for op, contexts in sorted(CONTEXT_TAXONOMY.items()):
        valid_contexts = [c for c in contexts if c != "invalid"]
        print(f"  {op}: {len(valid_contexts)} contexts - {valid_contexts}")
