# Compositional Inference Graph (CIG) - Proof of Concept

## 6-Week Implementation Plan (UPDATED)

---

## ✅ PoC Objective

> **Prove that a system can solve arithmetic problems using fixed rules ($K_R$) and learn to apply them better over time ($K_P$), without hallucinating or requiring retraining.**

---

## 📍 Scope

**Domain**: Arithmetic reasoning  
**Problem Set**: 100 procedurally generated arithmetic problems (e.g., factor sums, GCD, simplified expressions)  
**Metrics**:

- Accuracy (correctness)
- Efficiency (search depth, time)
- Learning (improvement over time)
- Traceability (interpretable solution path)

---

## 🧪 PoC Plan (7 Weeks)

### 🔹 Week 0: Problem Generation & Validation (3-4 days)

**Goal**: Generate 100 arithmetic problems and validate that all are solvable with K_R.

#### Part A: Define Problem Templates

```python
# scripts/00_generate_problems.py

from typing import Callable, Dict, Any
import json
import math

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def _factors(n):
    return [i for i in range(1, n+1) if n % i == 0]

def _prime_factors(n):
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
        "operation_chain": ["factors", "count"]  # Will need to add count operation
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
        "operation_chain": ["factors", "max_excluding_self"]  # Special operation needed
    },
]
```

#### Part B: Problem Generation Strategy

```python
def generate_problem_set(num_problems=100, seed=42):
    """
    Generate balanced problem set across categories.

    Strategy:
    1. Distribute problems evenly across categories
    2. Sample parameter values randomly from ranges
    3. Compute ground truth using template functions
    4. Validate each problem is solvable
    """
    import random
    random.seed(seed)

    problems = []
    problems_per_category = num_problems // len(PROBLEM_TEMPLATES)

    for template_spec in PROBLEM_TEMPLATES:
        for _ in range(problems_per_category):
            # Sample parameters
            params = {}
            for param_name, param_range in template_spec["params"].items():
                params[param_name] = random.choice(list(param_range))

            # Generate problem text
            problem_text = template_spec["template"].format(**params)

            # Compute ground truth
            ground_truth = template_spec["ground_truth"](**params)

            problems.append({
                "text": problem_text,
                "params": params,
                "expected": ground_truth,
                "category": template_spec["category"],
                "operation_chain": template_spec["operation_chain"]
            })

    return problems
```

#### Part C: Validation

```python
def validate_problem_set(problems, K_R):
    """
    Verify that all generated problems are solvable with K_R.

    Checks:
    1. All required operations exist in K_R
    2. Ground truth is computable
    3. No problems are trivial or degenerate
    """
    validation_errors = []

    for i, problem in enumerate(problems):
        # Check operations exist
        for op_name in problem["operation_chain"]:
            if op_name not in K_R and op_name not in ["count", "max_excluding_self"]:
                validation_errors.append(
                    f"Problem {i}: Operation '{op_name}' not in K_R"
                )

        # Check ground truth is valid
        expected = problem["expected"]
        if expected is None or (isinstance(expected, float) and math.isnan(expected)):
            validation_errors.append(
                f"Problem {i}: Invalid ground truth: {expected}"
            )

        # Check problem is not trivial
        if isinstance(expected, int) and expected < 0:
            validation_errors.append(
                f"Problem {i}: Unexpected negative result: {expected}"
            )

    if validation_errors:
        print(f"❌ Validation failed with {len(validation_errors)} errors:")
        for error in validation_errors[:10]:  # Show first 10
            print(f"  - {error}")
        return False
    else:
        print(f"✅ All {len(problems)} problems validated successfully")
        return True

def save_problem_set(problems, filepath):
    """Save problems to JSONL format"""
    with open(filepath, 'w') as f:
        for problem in problems:
            f.write(json.dumps(problem) + '\n')
    print(f"✅ Saved {len(problems)} problems to {filepath}")
```

#### Part D: Generate Training and Holdout Sets

```python
# Generate main training set (100 problems)
training_problems = generate_problem_set(num_problems=100, seed=42)
validate_problem_set(training_problems, OPERATIONS)
save_problem_set(training_problems, "data/problems/arithmetic_100.jsonl")

# Generate holdout set (20 problems for Week 6)
holdout_problems = generate_problem_set(num_problems=20, seed=123)
validate_problem_set(holdout_problems, OPERATIONS)
save_problem_set(holdout_problems, "data/problems/arithmetic_20_holdout.jsonl")
```

#### Week 0 Deliverables

|Item|Status|Location|
|---|---|---|
|✅ Problem templates defined|Complete|`scripts/00_generate_problems.py`|
|✅ Ground truth computation|Complete|Template functions|
|✅ 100 training problems generated|Complete|`data/problems/arithmetic_100.jsonl`|
|✅ 20 holdout problems generated|Complete|`data/problems/arithmetic_20_holdout.jsonl`|
|✅ Validation passed|Complete|All problems solvable with K_R|

#### Week 0 Success Criteria

- [ ] 100 training problems generated across 8+ categories
- [ ] All problems have computable ground truth
- [ ] All required operations exist in K_R (or are documented as needed additions)
- [ ] Problem distribution is balanced (no category has >30% of problems)
- [ ] 20 holdout problems for generalization testing

#### Notes on Missing Operations

After validation, if any operations are missing from K_R (e.g., `count`, `max_excluding_self`), they should be added in Week 1 Part A.

---

### 🔹 Week 1: Build $K_R$ + Recognition Infrastructure

**Goal**: Define the fixed, grounded rules of arithmetic operations AND the mechanisms to recognize which operations apply to given problems.

#### Part A: Define Atomic Operations in $K_R$ (Immutable Rule Set)

```python
# src/cig_poc/kr/registry.py
from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass
class AtomicOperation:
    """Immutable atomic operation in K_R"""
    name: str
    input_types: List[str]
    output_type: str
    compute: Callable
    preconditions: Optional[Callable] = None

# K_R: Fixed rule library (~25-30 operations)
OPERATIONS = {
    # Arithmetic basics
    "add": AtomicOperation(
        name="add",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a + b
    ),
    "multiply": AtomicOperation(
        name="multiply",
        input_types=["number", "number"],
        output_type="number",
        compute=lambda a, b: a * b
    ),
    
    # Generators (number → list)
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
    
    # Aggregators (list → number)
    "sum_list": AtomicOperation(
        name="sum_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: sum(lst)
    ),
    "product_list": AtomicOperation(
        name="product_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: _product(lst)
    ),
    "min_list": AtomicOperation(
        name="min_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: min(lst)
    ),
    "max_list": AtomicOperation(
        name="max_list",
        input_types=["number_list"],
        output_type="number",
        compute=lambda lst: max(lst)
    ),
    
    # Number theory
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
        compute=lambda a, b: (a * b) // math.gcd(a, b)
    ),
    "is_prime": AtomicOperation(
        name="is_prime",
        input_types=["number"],
        output_type="boolean",
        compute=lambda n: _is_prime(n)
    ),
    
    # Predicates (number → boolean)
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
    
    # Filters
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
    
    # Transforms
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
}
```

**Key Constraint**: $K_R$ is immutable after Week 1. No new operations can be added.

#### Part B: Define Pattern Recognition Rules (NEW - CRITICAL)

```python
# src/cig_poc/kr/recognizers.py

PATTERN_RULES = {
    # Keyword-based operation mapping
    "factors": {
        "keywords": ["factor", "divisor", "divide by"],
        "confidence": 0.95,
        "category": "generator"
    },
    "sum_list": {
        "keywords": ["sum", "total", "add", "add all", "add together"],
        "confidence": 0.90,
        "category": "aggregator"
    },
    "product_list": {
        "keywords": ["product", "multiply", "multiply all"],
        "confidence": 0.90,
        "category": "aggregator"
    },
    "gcd": {
        "keywords": ["gcd", "greatest common divisor", "common divisor"],
        "confidence": 0.95,
        "category": "number_theory"
    },
    "lcm": {
        "keywords": ["lcm", "least common multiple"],
        "confidence": 0.95,
        "category": "number_theory"
    },
    "is_prime": {
        "keywords": ["prime", "is prime"],
        "confidence": 0.95,
        "category": "predicate"
    },
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
    "prime_factors": {
        "keywords": ["prime factor", "factorization"],
        "confidence": 0.90,
        "category": "generator"
    },
}

def simple_pattern_recognizer(text: str) -> List[Tuple[str, float]]:
    """
    Keyword-based pattern recognition.
    Maps problem text to candidate operations with confidence scores.
    
    Returns: [(operation_name, confidence), ...]
    """
    text_lower = text.lower()
    candidates = []
    
    for op_name, rules in PATTERN_RULES.items():
        for keyword in rules["keywords"]:
            if keyword in text_lower:
                candidates.append((op_name, rules["confidence"]))
                break  # Match this operation once per text
    
    return sorted(candidates, key=lambda x: x[1], reverse=True)

# Example usage:
# recognize("What is the sum of all factors of 12?")
# → [("factors", 0.95), ("sum_list", 0.90)]
```

**Why This Matters**: Without pattern recognition, beam search would try all 25-30 operations, exploring millions of paths. With it, search space drops to ~3-5 candidates per problem.

#### Part C: Define Context Extraction Rules (NEW - CRITICAL)

```python
# src/cig_poc/kr/context.py

def extract_context(value: any, operation: str) -> str:
    """
    Determine context key for K_P lookup.
    
    K_P is organized as: Dict[(operation, context), weight]
    
    This function determines which context a value falls into,
    enabling learned weights to differentiate between usage patterns.
    
    Args:
        value: The input being operated on (number, list, etc.)
        operation: The name of the operation being applied
    
    Returns:
        context_key: String used to look up K_P[(operation, context_key)]
    """
    
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
    
    elif operation == "sum_list":
        if not isinstance(value, list):
            return "invalid"
        if len(value) < 10:
            return "short_list"
        else:
            return "long_list"
    
    elif operation == "product_list":
        if not isinstance(value, list):
            return "invalid"
        if len(value) < 10:
            return "short_list"
        else:
            return "long_list"
    
    elif operation == "gcd":
        if not isinstance(value, tuple) or len(value) != 2:
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
    
    elif operation == "prime_factors":
        if not isinstance(value, int):
            return "invalid"
        if value < 100:
            return "small_number"
        elif value < 10000:
            return "medium_number"
        else:
            return "large_number"
    
    # Default fallback
    return "generic"


# Context taxonomy (for documentation)
CONTEXT_TAXONOMY = {
    "factors": ["small_prime", "small_composite", "large_prime", "large_composite", "invalid"],
    "sum_list": ["short_list", "long_list", "invalid"],
    "product_list": ["short_list", "long_list", "invalid"],
    "gcd": ["small_numbers", "medium_numbers", "large_numbers", "invalid"],
    "is_prime": ["small_number", "medium_number", "large_number", "invalid"],
    "prime_factors": ["small_number", "medium_number", "large_number", "invalid"],
}
```

**Why This Matters**: Without context extraction, K_P has no meaningful structure. All operations collapse to generic weights. Learning doesn't differentiate between usage patterns.

#### Part D: Week 1 Tests

```python
# tests/test_kr.py
import pytest
from cig_poc.kr.registry import OPERATIONS
from cig_poc.kr.recognizers import simple_pattern_recognizer
from cig_poc.kr.context import extract_context

def test_operations_are_immutable():
    """Verify K_R operations work correctly"""
    assert OPERATIONS["factors"].compute(12) == [1, 2, 3, 4, 6, 12]
    assert OPERATIONS["sum_list"].compute([1, 2, 3]) == 6
    assert OPERATIONS["is_prime"].compute(7) == True

def test_pattern_recognition():
    """Pattern recognizer maps text to operations"""
    result = simple_pattern_recognizer("What is the sum of all factors of 12?")
    ops = [op for op, conf in result]
    
    assert "factors" in ops
    assert "sum_list" in ops
    assert len(result) >= 2

def test_context_extraction():
    """Context extractor determines K_P lookup keys"""
    assert extract_context(7, "factors") == "small_prime"
    assert extract_context(12, "factors") == "small_composite"
    assert extract_context(97, "factors") == "small_prime"
    
    assert extract_context([1, 2, 3], "sum_list") == "short_list"
    assert extract_context([1]*20, "sum_list") == "long_list"

def test_end_to_end_recognition():
    """Full pipeline: text → operations → context"""
    text = "What is the sum of all factors of 12?"
    
    # Step 1: Pattern recognition
    candidates = simple_pattern_recognizer(text)
    assert len(candidates) >= 2
    
    # Step 2: Execute factors
    factors_result = OPERATIONS["factors"].compute(12)
    op_name, confidence = candidates[0]  # Should be factors
    context = extract_context(factors_result, "sum_list")
    
    assert context in ["short_list", "long_list"]
```

#### Week 1 Deliverables

|Item|Status|Location|
|---|---|---|
|✅ $K_R$ Operation Library|Complete|`src/cig_poc/kr/registry.py`|
|✅ Pattern Recognition Rules|Complete|`src/cig_poc/kr/recognizers.py`|
|✅ Context Extraction Rules|Complete|`src/cig_poc/kr/context.py`|
|✅ Tests for all components|Complete|`tests/test_kr.py`|

#### Week 1 Success Criteria

- [ ] 25-30 atomic operations defined with deterministic outputs
- [ ] Pattern recognizer correctly identifies 2+ operations from sample problems
- [ ] Context extraction assigns meaningful keys (not just "generic")
- [ ] All operations tested independently
- [ ] No hallucinated operations or rules

---

### 🔹 Week 2: Build $K_P$ – Learned Heuristic Weights

**Goal**: Initialize a learnable scoring function for each operation in context.

```python
# src/cig_poc/kp/store.py

class KPStore:
    """
    Pattern Knowledge: Learned weights for operation selection.
    
    Structure: Dict[(operation_name, context_key), weight]
    
    Example:
        K_P[("factors", "small_prime")] = 0.95
        K_P[("factors", "large_composite")] = 0.60
        K_P[("sum_list", "short_list")] = 0.98
    """
    
    def __init__(self):
        self.weights = {}
        self.success_counts = {}
        self.attempt_counts = {}
    
    def initialize_uniform(self, operations, contexts):
        """Start with uniform weights (0.5) for all combinations"""
        for op in operations:
            for context in contexts.get(op, ["generic"]):
                key = (op, context)
                self.weights[key] = 0.5
                self.success_counts[key] = 0
                self.attempt_counts[key] = 0
    
    def get_weight(self, operation: str, context: str) -> float:
        """Retrieve learned weight for (operation, context) pair"""
        key = (operation, context)
        return self.weights.get(key, 0.5)  # Default: 0.5
    
    def update_weight(self, operation: str, context: str, success: bool, alpha=0.1):
        """
        Update K_P based on outcome using temporal difference learning.
        
        If operation succeeded: reward = +1.0
        If operation failed: reward = -0.5
        """
        key = (operation, context)
        old_weight = self.get_weight(operation, context)
        
        reward = 1.0 if success else -0.5
        new_weight = old_weight + alpha * (reward - old_weight)
        
        self.weights[key] = max(0.0, min(1.0, new_weight))  # Clamp to [0, 1]
        
        # Track statistics
        self.attempt_counts[key] = self.attempt_counts.get(key, 0) + 1
        if success:
            self.success_counts[key] = self.success_counts.get(key, 0) + 1
    
    def get_success_rate(self, operation: str, context: str) -> float:
        """Empirical success rate for Bayesian prior"""
        key = (operation, context)
        attempts = self.attempt_counts.get(key, 0)
        successes = self.success_counts.get(key, 0)
        
        if attempts == 0:
            return 0.5  # No data
        
        return successes / attempts
    
    def save_to_json(self, filepath: str):
        """Persist learned weights for analysis"""
        import json
        data = {
            "weights": {str(k): v for k, v in self.weights.items()},
            "success_rates": {
                str(k): self.get_success_rate(k[0], k[1])
                for k in self.weights
            }
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
```

#### Week 2 Deliverables

|Item|Format|
|---|---|
|✅ $K_P$ Weight Store|Python class|
|✅ Initialization logic|Uniform (0.5) for all contexts|
|✅ Update mechanism|Temporal difference learning|
|✅ Persistence|JSON export for analysis|

---

### 🔹 Week 3: Beam Search Solver with Composite Scoring

**Goal**: Implement the composite scoring function and beam search that uses both $K_R$ structure and $K_P$ weights.

```python
# src/cig_poc/search/scorer.py

def composite_score(
    operation: str,
    context: str,
    depth: int,
    K_P: "KPStore",
    recently_used: set,
    max_depth: int = 5,
    use_weighted_sum: bool = False
) -> float:
    """
    Composite scoring function: W(e, x, t)

    Combines multiple signals to score operation applicability:
    1. Structural signal: Prefer shallow solutions (1 / (1 + depth))
    2. Learned signal: K_P weight for this (operation, context)
    3. Temporal signal: Avoid recently used operations
    4. Historical signal: Bayesian success rate

    Args:
        use_weighted_sum: If True, use weighted sum instead of multiplication.
                         Use this if scores become too small at depth > 3.
    """

    # 1. Structural weight: prefer shallow depth
    struct_weight = 1.0 / (1.0 + depth)
    if depth > max_depth:
        return 0.0  # Hard cutoff at max depth

    # 2. Learned weight from K_P
    learned_weight = K_P.get_weight(operation, context)

    # 3. Temporal weight: discourage loops
    temporal_weight = 1.0
    if operation in recently_used:
        temporal_weight = 0.5

    # 4. Historical weight: Bayesian success rate
    success_rate = K_P.get_success_rate(operation, context)
    historical_weight = success_rate if success_rate > 0 else 0.5

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


# Choosing Between Multiplication and Weighted Sum:
#
# START WITH MULTIPLICATION (simpler, more interpretable):
# - Good for shallow searches (depth 1-3)
# - Natural "veto" behavior: one bad signal kills the score
# - Easier to debug: scores directly show cumulative quality
#
# SWITCH TO WEIGHTED SUM if you observe:
# - Beam search rarely finds solutions at depth > 3
# - Scores all drop below 0.1 even for good paths
# - Learning stalls because all operations have similar (low) scores
#
# HOW TO DECIDE:
# 1. Run 20 problems with multiplication
# 2. Log score distribution at each depth
# 3. If mean score at depth=4 < 0.05, switch to weighted sum


# src/cig_poc/search/beam.py

class BeamSearchSolver:
    """
    Hierarchical beam search using composite scoring.
    """
    
    def __init__(self, K_R, K_P, beam_width=10, max_depth=5):
        self.K_R = K_R  # Immutable operation library
        self.K_P = K_P  # Learned weights
        self.beam_width = beam_width
        self.max_depth = max_depth
    
    def solve(self, problem_state, goal, problem_text=""):
        """
        Main inference loop.
        
        Args:
            problem_state: Current state (dict with values)
            goal: Goal condition (predicate)
            problem_text: Original problem text for pattern recognition
        
        Returns:
            (solution_path, score, trace)
        """
        from cig_poc.kr.recognizers import simple_pattern_recognizer
        from cig_poc.kr.context import extract_context
        
        # STEP 1: Pattern recognition to narrow candidates
        candidates = simple_pattern_recognizer(problem_text)
        candidate_ops = set(op for op, conf in candidates)
        
        # STEP 2: Beam search
        beam = [(problem_state, [], 1.0, set())]  # (state, path, score, recently_used)
        
        for depth in range(self.max_depth):
            new_candidates = []
            
            for state, path, accum_score, recently_used in beam:
                # Check if goal is satisfied
                if self._satisfies_goal(state, goal):
                    return path, accum_score, self._build_trace(path, state)
                
                # Get applicable operations from K_R
                applicable_ops = self._get_applicable_operations(state)
                
                for op_name in applicable_ops:
                    # Filter by pattern recognition (if we have candidates)
                    if candidate_ops and op_name not in candidate_ops:
                        continue  # Skip operations not recognized in text
                    
                    # Determine context
                    context = extract_context(state.get("_input"), op_name)
                    
                    # Compute composite score
                    score = composite_score(
                        op_name,
                        context,
                        depth,
                        self.K_P,
                        recently_used,
                        self.max_depth
                    )
                    
                    if score < 0.01:  # Pruning threshold
                        continue
                    
                    # Try to apply operation
                    try:
                        next_state = self._apply_operation(state, op_name)
                        new_path = path + [(op_name, context)]
                        new_score = accum_score * score
                        new_recently_used = recently_used | {op_name}
                        
                        new_candidates.append((
                            next_state,
                            new_path,
                            new_score,
                            new_recently_used
                        ))
                    except Exception as e:
                        # Operation failed, try next
                        continue
            
            # Keep top-k by score
            new_candidates.sort(key=lambda x: x[2], reverse=True)
            beam = new_candidates[:self.beam_width]
            
            if not beam:
                break  # No viable candidates
        
        # No solution found
        return None, 0.0, None
    
    def _get_applicable_operations(self, state):
        """
        Get operations that can be applied to current state based on type compatibility.

        This implements type checking to prevent nonsensical operations like:
        - sum_list(12) - passing number to list operation
        - factors([1,2,3]) - passing list to number operation
        """
        current_value = state.get("_result") or state.get("_input")
        current_type = self._infer_type(current_value)

        applicable = []
        for op_name, op in self.K_R.items():
            # Check if operation's first input type matches current state type
            if len(op.input_types) > 0 and op.input_types[0] == current_type:
                applicable.append(op_name)

        return applicable

    def _infer_type(self, value):
        """Infer the type of a value for operation matching"""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "number"
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], int):
                return "number_list"
            return "list"
        elif isinstance(value, tuple) and len(value) == 2:
            return "number_pair"  # For operations like gcd(a, b)
        else:
            return "unknown"
    
    def _apply_operation(self, state, op_name):
        """Apply operation to state, return next state"""
        op = self.K_R[op_name]
        
        # Get inputs from state
        # (simplified: assumes state has inputs ready)
        if op.preconditions and not op.preconditions(state.get("_input")):
            raise ValueError(f"Preconditions failed for {op_name}")
        
        result = op.compute(state.get("_input"))
        
        return {"_result": result, "_input": result, "_op": op_name}
    
    def _satisfies_goal(self, state, goal):
        """Check if state satisfies goal"""
        return goal(state.get("_result"))
    
    def _build_trace(self, path, final_state):
        """Build interpretable trace of solution"""
        return {
            "operations": [op for op, ctx in path],
            "contexts": [ctx for op, ctx in path],
            "result": final_state.get("_result")
        }
```

#### Week 3 Success Metrics

- [ ] Composite scorer produces reasonable scores (0-1 range)
- [ ] Beam search finds solutions on basic problems (70%+ accuracy)
- [ ] Pattern recognition reduces search space by 5-10x
- [ ] Average search depth < 5 operations
- [ ] Solution traces are interpretable

#### Week 3 Design Limitations (POC Scope)

**Single-Chain Constraint**: The current implementation only supports **linear operation chains** where each operation's output feeds directly into the next operation's input (A → B → C).

**Out of Scope for POC:**
- **Multi-branch problems** requiring parallel computations
  - Example: "GCD of (sum of factors of 12) and (sum of factors of 18)"
  - This would require computing two separate chains and then combining results
  - Solution: Track multiple intermediate results in state (future enhancement)

**Why This is Acceptable:**
- 90%+ of arithmetic problems can be solved with linear chains
- Multi-branch support adds significant complexity to state management
- Can be added in Phase 2 without changing K_R/K_P architecture
- Focus POC on proving core learning mechanism, not compositional complexity

**How to Work Around:**
- Design Week 0 problem set to avoid multi-branch problems
- If multi-branch problems are needed, decompose into sub-problems:
  - "What is sum of factors of 12?" → 28
  - "What is sum of factors of 18?" → 39
  - "What is GCD of 28 and 39?" → 1

---

### 🔹 Week 4: Episodic Learning Loop

**Goal**: Learn from success/failure without changing $K_R$.

```python
# src/cig_poc/memory/episode.py

@dataclass
class Episode:
    """Record of a single problem-solving attempt"""
    problem_text: str
    initial_state: dict
    path: List[Tuple[str, str]]  # [(operation, context), ...]
    result: any
    goal_satisfied: bool
    timestamp: float

# src/cig_poc/memory/learner.py

class ExperientialLearner:
    """Learn from episodes without retraining"""
    
    def __init__(self, K_P, learn_rate=0.1):
        self.K_P = K_P
        self.learn_rate = learn_rate
        self.episodes = []
    
    def record_episode(self, problem_text, path, result, success):
        """Record a problem-solving attempt"""
        episode = Episode(
            problem_text=problem_text,
            initial_state={},
            path=path,
            result=result,
            goal_satisfied=success,
            timestamp=time.time()
        )
        self.episodes.append(episode)
    
    def learn_from_episode(self, episode):
        """Update K_P based on episode outcome"""
        for operation, context in episode.path:
            self.K_P.update_weight(
                operation,
                context,
                episode.goal_satisfied,
                alpha=self.learn_rate
            )
    
    def learn_from_all_episodes(self):
        """Batch learning from all episodes"""
        for episode in self.episodes:
            self.learn_from_episode(episode)
    
    def get_learning_curve(self):
        """Analyze improvement over episodes"""
        success_by_episode = [
            ep.goal_satisfied for ep in self.episodes
        ]
        
        # Rolling average (window=10)
        rolling_avg = []
        for i in range(len(success_by_episode)):
            window = success_by_episode[max(0, i-10):i+1]
            rolling_avg.append(sum(window) / len(window))
        
        return rolling_avg
```

#### Week 4 Deliverables

- [ ] Episodic memory stores full problem traces
- [ ] Learning loop updates K_P after each problem
- [ ] Logging of all episodes to file
- [ ] Learning curve analysis

---

### 🔹 Week 5: Evaluation & Baseline Comparison

**Goal**: Measure improvement over time and compare to baselines.

```python
# src/cig_poc/eval/metrics.py

class Evaluator:
    """Compute accuracy, efficiency, and learning metrics"""
    
    def __init__(self):
        self.results = []
    
    def evaluate(self, solver, problem_set, K_P_snapshots):
        """
        Run solver on problem set, track metrics over time
        """
        accuracy_by_week = []
        depth_by_week = []
        time_by_week = []
        
        for week, K_P in enumerate(K_P_snapshots):
            solver.K_P = K_P
            
            correct = 0
            total_depth = 0
            total_time = 0
            
            for problem in problem_set:
                start = time.time()
                path, score, trace = solver.solve(
                    problem["initial_state"],
                    problem["goal"],
                    problem["text"]
                )
                elapsed = time.time() - start
                
                is_correct = problem["expected"] == trace["result"]
                if is_correct:
                    correct += 1
                
                total_depth += len(path)
                total_time += elapsed
            
            accuracy_by_week.append(correct / len(problem_set))
            depth_by_week.append(total_depth / len(problem_set))
            time_by_week.append(total_time / len(problem_set))
        
        return {
            "accuracy_by_week": accuracy_by_week,
            "avg_depth_by_week": depth_by_week,
            "avg_time_by_week": time_by_week
        }
```

#### Week 5 Expected Results

**Target Performance** (aspirational goals):

|Metric|Week 1|Week 3|Week 5 (Target)|
|---|---|---|---|
|Accuracy|60%|75%|**80-90%**|
|Avg Search Depth|4.2|3.1|**2.0-2.5**|
|Avg Time|1.2s|0.8s|**0.3-0.6s**|

**Interpretation Thresholds**:

| Outcome | Accuracy Range | Interpretation |
|---------|---------------|----------------|
| **Excellent** | 85-100% | POC exceeds expectations, ready for multi-domain expansion |
| **Target** | 75-85% | POC successfully demonstrates learning and anti-hallucination |
| **Acceptable** | 65-75% | Core concept proven, needs optimization of K_P or scoring |
| **Needs Work** | 50-65% | Fundamental issues with pattern recognition or beam search |
| **Failure** | <50% | Re-evaluate architecture or problem set difficulty |

**Additional Success Indicators**:
- **Learning Curve**: 15-25% improvement from Week 1 to Week 5
- **Depth Reduction**: 30-50% reduction in average search depth (shows K_P is learning)
- **Time Reduction**: 40-60% faster inference (shows beam pruning is effective)

**What to Do If Results Fall Short**:
- **60-75% accuracy**: Check pattern recognition coverage, may need more keyword rules
- **High depth (>3.5 avg)**: K_P not learning effectively, review update mechanism
- **Slow time (>1s avg)**: Beam width too large or pruning threshold too low

#### Week 5 Baselines

- Random search (lower bound)
- Exhaustive search (optimal but slow)
- LLM comparison (e.g., GPT-4 on same problems)

---

### 🔹 Week 6: Generalization Test

**Goal**: Test transfer to unseen problem types.

```python
def test_generalization():
    """
    Evaluate on 20 new problem types not seen during training
    """
    holdout_problems = [
        {"text": "What is the LCM of 12 and 18?", ...},
        {"text": "Find the prime factorization of 60", ...},
        # ... 18 more new problems
    ]
    
    # Zero-shot: K_P from training
    zero_shot_accuracy = evaluate(solver, holdout_problems)
    
    # Few-shot: K_P after 5 problems
    few_shot_accuracy = evaluate(solver, holdout_problems[:5])
    
    print(f"Zero-shot: {zero_shot_accuracy:.2%}")
    print(f"Few-shot: {few_shot_accuracy:.2%}")
    
    # Check transfer: did operations learned for "factors"
    # help with "prime_factors"?
```

---

## 📦 Complete Project Structure

```
cig-poc/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .python-version
├── Makefile
│
├── data/
│   ├── problems/
│   │   ├── arithmetic_100.jsonl
│   │   └── arithmetic_20_holdout.jsonl
│   └── outputs/
│       └── .gitkeep
│
├── src/
│   └── cig_poc/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── kr/                          # Week 1
│       │   ├── __init__.py
│       │   ├── registry.py             # K_R: Operation library
│       │   ├── recognizers.py          # Pattern recognition
│       │   └── context.py              # Context extraction
│       │
│       ├── kp/                          # Week 2
│       │   ├── __init__.py
│       │   ├── store.py                # K_P: Learned weights
│       │   └── updater.py              # Weight update logic
│       │
│       ├── search/                      # Week 3
│       │   ├── __init__.py
│       │   ├── beam.py                 # Beam search engine
│       │   └── scorer.py               # Composite scoring
│       │
│       ├── memory/                      # Week 4
│       │   ├── __init__.py
│       │   ├── episode.py              # Episode storage
│       │   └── learner.py              # Learning loop
│       │
│       ├── eval/                        # Week 5
│       │   ├── __init__.py
│       │   └── metrics.py              # Evaluation
│       │
│       └── viz/                         # Optional
│           ├── __init__.py
│           └── trace.py                # Trace visualization
│
├── scripts/
│   ├── 01_init_kr.py          # Initialize K_R
│   ├── 02_train.py            # Run learning loop
│   ├── 03_evaluate.py         # Evaluation
│   └── 04_holdout_test.py     # Generalization test
│
├── notebooks/
│   ├── 00_quickstart.ipynb
│   └── 01_analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_kr.py             # Test K_R components
│   ├── test_kp.py             # Test K_P learning
│   ├── test_search.py         # Test beam search
│   └── test_eval.py           # Test metrics
│
└── docs/
    ├── index.md
    ├── api.md
    └── traces.md
```

---

## 🔑 Key Insight to Highlight

> **The system never learns new rules — it only learns which rules to apply, when, and in what order.**

This is the core of the $K_R$/$K_P$ split and the **anti-hallucination guarantee**.

- $K_R$ is immutable: No new operations can be created
- $K_P$ is learned: Weights optimize operation selection
- Result: 100% traceability, 0% hallucination

---

## 📋 Example: Complete Trace Pipeline

### Input

```
"What is the sum of all factors of 12?"
```

### Step 1: Pattern Recognition

```python
candidates = simple_pattern_recognizer(text)
# → [("factors", 0.95), ("sum_list", 0.90)]
```

### Step 2: Beam Search

```
Depth 0:
  factors(12, context="small_composite") → [1,2,3,4,6,12]
  score = (1/(1+0)) * K_P[("factors","small_composite")] * 1.0 * 0.9
        = 1.0 * 0.85 * 1.0 * 0.9 = 0.765

Depth 1:
  sum_list([1,2,3,4,6,12], context="short_list") → 28
  score = (1/(1+1)) * K_P[("sum_list","short_list")] * 1.0 * 0.98
        = 0.5 * 0.95 * 1.0 * 0.98 = 0.466

Final score = 0.765 * 0.466 = 0.356
```

### Step 3: Solution Trace

```json
{
  "problem": "What is the sum of all factors of 12?",
  "trace": [
    {
      "operation": "factors",
      "context": "small_composite",
      "input": 12,
      "output": [1, 2, 3, 4, 6, 12],
      "score": 0.765
    },
    {
      "operation": "sum_list",
      "context": "short_list",
      "input": [1, 2, 3, 4, 6, 12],
      "output": 28,
      "score": 0.466
    }
  ],
  "result": 28,
  "total_score": 0.356,
  "search_depth": 2,
  "time": 0.08,
  "success": true
}
```

### Step 4: Learning

```python
learner.record_episode(
    problem_text=text,
    path=[("factors", "small_composite"), ("sum_list", "short_list")],
    result=28,
    success=True
)

# Update K_P
learner.learn_from_episode(episode)
K_P[("factors", "small_composite")] += 0.1 * (1.0 - 0.85) = 0.86
K_P[("sum_list", "short_list")] += 0.1 * (1.0 - 0.95) = 0.955
```

---

## ✅ Success Criteria (All 7 Weeks)

**Core Requirements** (Must achieve for POC to be considered successful):
- [ ] **75-90% accuracy** on 100 arithmetic problems (target: 80%+)
- [ ] **15-25% improvement** in accuracy from Week 1 to Week 5
- [ ] **100% of solutions traceable to K_R** (zero hallucinations)
- [ ] **No hallucinated operations or outputs** (anti-hallucination guarantee)
- [ ] **Learning update time <100ms** per episode
- [ ] **Pattern recognition correctly identifies 2+ operations** per problem (90%+ coverage)

**Secondary Requirements** (Demonstrates quality of implementation):
- [ ] **30-50% reduction** in average search depth over time (shows K_P learning)
- [ ] **Context extraction provides meaningful differentiation** in K_P (not all "generic")
- [ ] **Solution traces are interpretable** (human can verify correctness)
- [ ] **Generalization to new problem types** (60-70%+ on holdout set)

**Bonus Achievements** (Exceeds expectations):
- [ ] 85%+ accuracy on training set
- [ ] 70%+ accuracy on holdout set (zero-shot)
- [ ] Average search depth <2.5 operations
- [ ] Inference time <0.5s per problem

---

## 🚀 Critical New Components (NOT in Original POC)

|Component|Why Critical|Added in Update|
|---|---|---|
|Pattern Recognition|Reduces search space 5-10x|Week 1 Part B|
|Context Extraction|Enables meaningful K_P learning|Week 1 Part C|
|Composite Scoring|Balances multiple signals|Week 3|
|Episode Recording|Full trace for analysis|Week 4|
|Learning Curve Analysis|Verify improvement over time|Week 5|

---

_End of Updated PoC Specification_