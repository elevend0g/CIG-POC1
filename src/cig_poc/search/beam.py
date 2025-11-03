"""
Week 3: Beam Search Solver

Hierarchical beam search using composite scoring to find solutions.
Combines pattern recognition, type checking, and learned weights.
"""

from typing import Dict, List, Tuple, Optional, Callable, Set, Any
from cig_poc.search.scorer import composite_score
from cig_poc.kr.recognizers import simple_pattern_recognizer, extract_numbers
from cig_poc.kr.context import extract_context


class BeamSearchSolver:
    """
    Hierarchical beam search using composite scoring.

    Key Features:
        - Pattern recognition to narrow candidate operations
        - Type checking to prevent invalid operation chains
        - Composite scoring combining structural, learned, temporal, and historical signals
        - Beam width limits search space (default: 10)
        - Max depth prevents infinite loops (default: 5)
    """

    def __init__(self, K_R: Dict, K_P, beam_width: int = 10, max_depth: int = 5):
        """
        Initialize beam search solver.

        Args:
            K_R: Immutable operation library
            K_P: Learned weights store
            beam_width: Maximum number of paths to keep at each depth
            max_depth: Maximum search depth
        """
        self.K_R = K_R
        self.K_P = K_P
        self.beam_width = beam_width
        self.max_depth = max_depth

    def solve(
        self,
        problem_text: str,
        initial_value: Any,
        goal: Callable[[Any], bool]
    ) -> Tuple[Optional[List[Tuple[str, str]]], float, Optional[Dict]]:
        """
        Main inference loop using beam search.

        Args:
            problem_text: Original problem text for pattern recognition
            initial_value: Starting value (e.g., a number or list)
            goal: Goal predicate that returns True when solution is found

        Returns:
            Tuple of (solution_path, final_score, trace)
                - solution_path: List of (operation, context) tuples, or None if not found
                - final_score: Accumulated score of solution
                - trace: Interpretable trace with operations and result

        Example:
            >>> solver = BeamSearchSolver(K_R, K_P)
            >>> path, score, trace = solver.solve(
            ...     "What is the sum of all factors of 12?",
            ...     12,
            ...     lambda x: x == 28  # Expected answer
            ... )
        """
        # STEP 1: Pattern recognition to narrow candidates
        candidates = simple_pattern_recognizer(problem_text)
        candidate_ops = {op for op, conf in candidates} if candidates else set()

        if not candidate_ops:
            # If no candidates, allow all operations (fallback)
            candidate_ops = set(self.K_R.keys())

        # STEP 2: Initialize beam with initial state
        # Each beam entry: (state, path, score, recently_used)
        initial_state = {"_input": initial_value, "_result": None}
        beam = [(initial_state, [], 1.0, set())]

        # Track statistics
        self.search_stats = {
            "nodes_explored": 0,
            "max_beam_size": 0,
            "depth_reached": 0
        }

        # STEP 3: Beam search
        for depth in range(self.max_depth):
            new_candidates = []

            for state, path, accum_score, recently_used in beam:
                self.search_stats["nodes_explored"] += 1

                # Check if goal is satisfied
                current_value = state.get("_result") if state.get("_result") is not None else state.get("_input")
                if goal(current_value):
                    self.search_stats["depth_reached"] = depth
                    return path, accum_score, self._build_trace(path, current_value)

                # Get applicable operations from K_R based on type checking
                applicable_ops = self._get_applicable_operations(state)

                for op_name in applicable_ops:
                    # Filter by pattern recognition (if we have candidates)
                    if candidate_ops and op_name not in candidate_ops:
                        continue  # Skip operations not recognized in text

                    # Get current value for context extraction
                    current_value = state.get("_result") if state.get("_result") is not None else state.get("_input")

                    # Determine context
                    context = extract_context(current_value, op_name)

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
                        # Operation failed (preconditions, type error, etc.)
                        continue

            # Keep top-k by score
            new_candidates.sort(key=lambda x: x[2], reverse=True)
            beam = new_candidates[:self.beam_width]

            self.search_stats["max_beam_size"] = max(self.search_stats["max_beam_size"], len(beam))

            if not beam:
                break  # No viable candidates

        # Check final beam for solutions
        for state, path, accum_score, recently_used in beam:
            current_value = state.get("_result") if state.get("_result") is not None else state.get("_input")
            if goal(current_value):
                self.search_stats["depth_reached"] = len(path)
                return path, accum_score, self._build_trace(path, current_value)

        # No solution found
        return None, 0.0, None

    def _get_applicable_operations(self, state: Dict) -> List[str]:
        """
        Get operations that can be applied to current state based on type compatibility.

        This implements type checking to prevent nonsensical operations like:
        - sum_list(12) - passing number to list operation
        - factors([1,2,3]) - passing list to number operation

        Args:
            state: Current state dictionary

        Returns:
            List of operation names that can be applied
        """
        current_value = state.get("_result") if state.get("_result") is not None else state.get("_input")
        current_type = self._infer_type(current_value)

        applicable = []
        for op_name, op in self.K_R.items():
            # Check if operation's first input type matches current state type
            if len(op.input_types) > 0:
                # For operations with multiple inputs (like gcd, add), need special handling
                if len(op.input_types) == 2 and op.input_types[0] == "number" and op.input_types[1] == "number":
                    # Skip binary operations in POC (would need state to track multiple values)
                    continue

                if op.input_types[0] == current_type:
                    applicable.append(op_name)

        return applicable

    def _infer_type(self, value: Any) -> str:
        """
        Infer the type of a value for operation matching.

        Args:
            value: Value to check type of

        Returns:
            Type string: "number", "number_list", "boolean", etc.
        """
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

    def _apply_operation(self, state: Dict, op_name: str) -> Dict:
        """
        Apply operation to state, return next state.

        Args:
            state: Current state
            op_name: Operation to apply

        Returns:
            New state dictionary

        Raises:
            ValueError: If preconditions not met
            Exception: If operation fails
        """
        op = self.K_R[op_name]

        # Get input from state
        current_value = state.get("_result") if state.get("_result") is not None else state.get("_input")

        # Check preconditions if any
        if op.preconditions and not op.preconditions(current_value):
            raise ValueError(f"Preconditions failed for {op_name}")

        # Apply operation
        result = op.compute(current_value)

        # Return new state
        return {
            "_input": state.get("_input"),  # Keep original input
            "_result": result,
            "_op": op_name
        }

    def _build_trace(self, path: List[Tuple[str, str]], final_result: Any) -> Dict:
        """
        Build interpretable trace of solution.

        Args:
            path: List of (operation, context) tuples
            final_result: Final result value

        Returns:
            Trace dictionary
        """
        return {
            "operations": [op for op, ctx in path],
            "contexts": [ctx for op, ctx in path],
            "path": path,
            "result": final_result,
            "depth": len(path)
        }


if __name__ == "__main__":
    # Test beam search solver
    print("=" * 60)
    print("Beam Search Solver Tests")
    print("=" * 60)

    # Import required modules
    import sys
    sys.path.insert(0, "/home/jay/ag0/cig/src")
    from cig_poc.kr.registry import OPERATIONS
    from cig_poc.kr.context import CONTEXT_TAXONOMY
    from cig_poc.kp.store import KPStore

    # Initialize K_P
    kp = KPStore()
    kp.initialize_uniform(OPERATIONS, CONTEXT_TAXONOMY)

    # Create solver
    solver = BeamSearchSolver(OPERATIONS, kp, beam_width=10, max_depth=5)

    # Test problem: "What is the sum of all factors of 12?"
    # Expected: factors(12) = [1,2,3,4,6,12], sum([1,2,3,4,6,12]) = 28
    print("\n🧪 Test Problem: Sum of all factors of 12")
    print("Expected: 28")

    path, score, trace = solver.solve(
        problem_text="What is the sum of all factors of 12?",
        initial_value=12,
        goal=lambda x: x == 28
    )

    if path:
        print(f"\n✅ Solution found!")
        print(f"  Operations: {trace['operations']}")
        print(f"  Contexts: {trace['contexts']}")
        print(f"  Result: {trace['result']}")
        print(f"  Depth: {trace['depth']}")
        print(f"  Score: {score:.4f}")
        print(f"\n  Search stats:")
        print(f"    Nodes explored: {solver.search_stats['nodes_explored']}")
        print(f"    Max beam size: {solver.search_stats['max_beam_size']}")
    else:
        print(f"\n❌ No solution found")

    # Test problem 2: "How many factors does 24 have?"
    # Expected: factors(24) = [1,2,3,4,6,8,12,24], count([...]) = 8
    print("\n" + "=" * 60)
    print("🧪 Test Problem 2: How many factors does 24 have?")
    print("Expected: 8")

    path, score, trace = solver.solve(
        problem_text="How many factors does 24 have?",
        initial_value=24,
        goal=lambda x: x == 8
    )

    if path:
        print(f"\n✅ Solution found!")
        print(f"  Operations: {trace['operations']}")
        print(f"  Contexts: {trace['contexts']}")
        print(f"  Result: {trace['result']}")
        print(f"  Depth: {trace['depth']}")
        print(f"  Score: {score:.4f}")
    else:
        print(f"\n❌ No solution found")
