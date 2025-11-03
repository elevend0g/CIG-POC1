"""
Week 1 Part A: K_R - Immutable Atomic Operations Registry

This module defines the fixed rule library (K_R) for the CIG PoC.
After Week 1, this registry is FROZEN - no new operations can be added.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Any
import math

@dataclass
class AtomicOperation:
    """
    Immutable atomic operation in K_R.

    Attributes:
        name: Unique identifier for the operation
        input_types: Expected input types (e.g., ["number"], ["number", "number"], ["number_list"])
        output_type: Type produced by this operation (e.g., "number", "boolean", "number_list")
        compute: Callable that performs the actual computation
        preconditions: Optional callable that validates inputs before execution
    """
    name: str
    input_types: List[str]
    output_type: str
    compute: Callable
    preconditions: Optional[Callable] = None

    def __call__(self, *args):
        """Execute the operation with given arguments"""
        if self.preconditions and not self.preconditions(*args):
            raise ValueError(f"Preconditions not met for {self.name}")
        return self.compute(*args)


# Helper functions for complex operations
def _is_prime(n: int) -> bool:
    """Check if n is prime"""
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _prime_factors(n: int) -> List[int]:
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

def _product(lst: List[int]) -> int:
    """Compute product of list elements"""
    result = 1
    for x in lst:
        result *= x
    return result


# K_R: Fixed rule library (~30 operations)
# IMMUTABLE AFTER WEEK 1 - NO NEW OPERATIONS CAN BE ADDED
OPERATIONS = {
    # ========== Arithmetic Basics ==========
    "add": AtomicOperation(
        name="add",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a + b
    ),
    "subtract": AtomicOperation(
        name="subtract",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a - b
    ),
    "multiply": AtomicOperation(
        name="multiply",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a * b
    ),
    "divide": AtomicOperation(
        name="divide",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a // b,
        preconditions=lambda a, b: b != 0
    ),
    "modulo": AtomicOperation(
        name="modulo",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a % b,
        preconditions=lambda a, b: b != 0
    ),

    # ========== Generators (number → list) ==========
    "factors": AtomicOperation(
        name="factors",
        input_types=["number"],
        output_type="number_list",
        compute=lambda n: [i for i in range(1, n+1) if n % i == 0]
    ),
    "prime_factors": AtomicOperation(
        name="prime_factors",
        input_types=["number"],
        output_type="number_list",
        compute=lambda n: _prime_factors(n)
    ),
    "range_to_n": AtomicOperation(
        name="range_to_n",
        input_types=["number"],
        output_type="number_list",
        compute=lambda n: list(range(1, n+1))
    ),

    # ========== Aggregators (list → number) ==========
    "sum_list": AtomicOperation(
        name="sum_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: sum(lst) if lst else 0
    ),
    "product_list": AtomicOperation(
        name="product_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: _product(lst) if lst else 1
    ),
    "min_list": AtomicOperation(
        name="min_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: min(lst) if lst else None,
        preconditions=lambda lst: len(lst) > 0
    ),
    "max_list": AtomicOperation(
        name="max_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: max(lst) if lst else None,
        preconditions=lambda lst: len(lst) > 0
    ),
    "count": AtomicOperation(
        name="count",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: len(lst)
    ),
    "average": AtomicOperation(
        name="average",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: sum(lst) // len(lst) if lst else 0,
        preconditions=lambda lst: len(lst) > 0
    ),

    # ========== Number Theory ==========
    "gcd": AtomicOperation(
        name="gcd",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: math.gcd(a, b)
    ),
    "lcm": AtomicOperation(
        name="lcm",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: (a * b) // math.gcd(a, b)  # Correct LCM formula
    ),
    "is_prime": AtomicOperation(
        name="is_prime",
        input_types=["number"],
        output_type="boolean",
        compute=lambda n: _is_prime(n)
    ),

    # ========== Predicates (number → boolean) ==========
    "is_even": AtomicOperation(
        name="is_even",
        input_types=["number"],
        output_type="boolean",
        compute=lambda n: n % 2 == 0
    ),
    "is_odd": AtomicOperation(
        name="is_odd",
        input_types=["number"],
        output_type="boolean",
        compute=lambda n: n % 2 == 1
    ),
    "greater_than": AtomicOperation(
        name="greater_than",
        input_types=["number", "number"],
        output_type="boolean",
        compute=lambda a, b: a > b
    ),
    "less_than": AtomicOperation(
        name="less_than",
        input_types=["number", "number"],
        output_type="boolean",
        compute=lambda a, b: a < b
    ),
    "equals": AtomicOperation(
        name="equals",
        input_types=["number", "number"],
        output_type="boolean",
        compute=lambda a, b: a == b
    ),

    # ========== Filters (list → list) ==========
    "filter_even": AtomicOperation(
        name="filter_even",
        input_types=["number_list"],
        output_type="number_list",
        compute=lambda lst: [x for x in lst if x % 2 == 0]
    ),
    "filter_odd": AtomicOperation(
        name="filter_odd",
        input_types=["number_list"],
        output_type="number_list",
        compute=lambda lst: [x for x in lst if x % 2 == 1]
    ),
    "filter_primes": AtomicOperation(
        name="filter_primes",
        input_types=["number_list"],
        output_type="number_list",
        compute=lambda lst: [x for x in lst if _is_prime(x)]
    ),

    # ========== Transforms (number → number) ==========
    "square": AtomicOperation(
        name="square",
        input_types=["number"],
        output_type="number",
        compute=lambda n: n * n
    ),
    "double": AtomicOperation(
        name="double",
        input_types=["number"],
        output_type="number",
        compute=lambda n: n * 2
    ),
    "half": AtomicOperation(
        name="half",
        input_types=["number"],
        output_type="number",
        compute=lambda n: n // 2 if n % 2 == 0 else None,
        preconditions=lambda n: n % 2 == 0
    ),
    "increment": AtomicOperation(
        name="increment",
        input_types=["number"],
        output_type="number",
        compute=lambda n: n + 1
    ),
    "decrement": AtomicOperation(
        name="decrement",
        input_types=["number"],
        output_type="number",
        compute=lambda n: n - 1
    ),

    # ========== Special Operations ==========
    "max_excluding_self": AtomicOperation(
        name="max_excluding_self",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: lst[-2] if len(lst) > 1 else (lst[0] if len(lst) == 1 else None),
        preconditions=lambda lst: len(lst) > 0
    ),
    "first": AtomicOperation(
        name="first",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: lst[0] if lst else None,
        preconditions=lambda lst: len(lst) > 0
    ),
    "last": AtomicOperation(
        name="last",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: lst[-1] if lst else None,
        preconditions=lambda lst: len(lst) > 0
    ),
}

# Validation: Ensure all operations are properly defined
def validate_kr():
    """Validate that K_R is properly constructed"""
    errors = []

    for op_name, op in OPERATIONS.items():
        if op.name != op_name:
            errors.append(f"Operation {op_name} has mismatched name {op.name}")
        if not op.input_types:
            errors.append(f"Operation {op_name} has no input types")
        if not op.output_type:
            errors.append(f"Operation {op_name} has no output type")
        if not op.compute:
            errors.append(f"Operation {op_name} has no compute function")

    if errors:
        raise ValueError(f"K_R validation failed:\n" + "\n".join(errors))

    print(f"✅ K_R validated: {len(OPERATIONS)} operations defined")
    return True

# Auto-discovered abstraction hierarchy
ABSTRACTION_HIERARCHY = {
    "generators": [op for op in OPERATIONS.values() if op.output_type == "number_list"],
    "aggregators": [op for op in OPERATIONS.values()
                   if "number_list" in op.input_types and op.output_type == "number"],
    "filters": [op for op in OPERATIONS.values()
               if "number_list" in op.input_types and op.output_type == "number_list"],
    "transforms": [op for op in OPERATIONS.values()
                  if op.input_types == ["number"] and op.output_type == "number"],
    "predicates": [op for op in OPERATIONS.values() if op.output_type == "boolean"],
    "arithmetic": [op for op in OPERATIONS.values()
                  if len(op.input_types) == 2 and op.input_types[0] == "number"
                  and op.input_types[1] == "number" and op.output_type == "number"],
}

if __name__ == "__main__":
    # Validate K_R on module load
    validate_kr()

    print("\n📊 K_R Statistics:")
    print(f"  Total operations: {len(OPERATIONS)}")
    print(f"\nAbstraction hierarchy:")
    for category, ops in ABSTRACTION_HIERARCHY.items():
        print(f"  - {category}: {len(ops)} operations")

    # Test a few operations
    print("\n🧪 Sample Operations:")
    print(f"  factors(12) = {OPERATIONS['factors'](12)}")
    print(f"  sum_list([1,2,3,4]) = {OPERATIONS['sum_list']([1,2,3,4])}")
    print(f"  is_prime(17) = {OPERATIONS['is_prime'](17)}")
    print(f"  lcm(12, 18) = {OPERATIONS['lcm'](12, 18)}")
    print(f"  gcd(12, 18) = {OPERATIONS['gcd'](12, 18)}")
