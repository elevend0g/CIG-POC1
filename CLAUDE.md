# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements the Compositional Inference Graph (CIG), a novel AI architecture that addresses fundamental limitations of Large Language Models through a hybrid approach combining symbolic reasoning with learned composition patterns. The system operates on well-defined atomic operations that can be composed, learned, and traced explicitly rather than relying on massive parameter matrices.

## Core Architecture Concepts

### Fundamental Components

The CIG system is built on five core components:

1. **Atomic Operations** (`AtomicOperation`): Fundamental units of computation with clear input/output contracts, embeddings, and success contexts
2. **Pattern Recognition Layer** (`MultiModalPatternRecognizer`): Maps text to applicable operations using keyword matching, semantic similarity, and learned patterns
3. **Compositional Inference Engine** (`CompositionEngine`): Hierarchical beam search through operation space to find solutions
4. **Experiential Learning System** (`ExperientialLearner`): Episodic memory and contextual weight updates without rule induction
5. **Abstraction Hierarchy** (`AbstractionHierarchy`): Auto-discovered operation categories (transforms, aggregators, generators, predicates)

### Key Design Principles

- **Operations as First-Class Citizens**: The system recognizes which tools apply and learns successful compositions rather than attempting to understand natural language
- **Multi-Strategy Recognition**: Multiple recognizers vote (keywords 0.8 confidence, semantic 0.7, learned 0.9) to avoid single points of failure
- **Hierarchical Planning**: Abstract operations reduce search space by 100-1000x before concrete sequence generation
- **No Rule Induction**: Similarity-based retrieval from episodic memory instead of symbolic rules

## Development Phases

The project follows a phased implementation strategy outlined in `docs/POC-updated.md`:

### Proof of Concept (7 Weeks)

**Week 0: Problem Generation & Validation** (3-4 days)
- Generate 100 procedurally generated arithmetic problems with ground truth
- Validate all problems are solvable with K_R
- Create 20 holdout problems for generalization testing

**Week 1: K_R + Recognition Infrastructure**
- Define 25-30 atomic operations (immutable rule set)
- Implement keyword-based pattern recognition
- Build context extraction rules for K_P
- Comprehensive testing of all components

**Week 2: K_P - Learned Heuristic Weights**
- Initialize learnable weight store: Dict[(operation, context), weight]
- Implement temporal difference learning update mechanism
- JSON persistence for analysis

**Week 3: Beam Search with Composite Scoring**
- Hierarchical beam search (width=10, max_depth=5)
- Composite scoring combining structural, learned, temporal, and historical signals
- Type checking to prevent invalid operation chains
- Note: POC limited to single-chain problems (linear compositions)

**Week 4: Episodic Learning Loop**
- Episodic memory for full problem traces
- Automatic K_P updates after each episode
- Learning curve analysis with rolling averages

**Week 5: Evaluation & Baselines**
- Target: 75-90% accuracy (80%+ ideal)
- Measure learning improvement (15-25% expected)
- Compare to random search, exhaustive search, and LLM baselines

**Week 6: Generalization Testing**
- Test on 20 holdout problems
- Zero-shot performance: 60-70%+ expected
- Measure transfer of K_P weights across problem types

### Future Phases (Post-POC)

**Phase 2: Multi-Domain Expansion**
- Expand to logic puzzles, simple planning, basic algebra
- Domain detection and cross-domain transfer
- Multi-branch problem support (parallel compositions)

**Phase 3: Scale Testing**
- Distributed operation execution
- Parallel beam search
- Hierarchical abstraction layers

## Technical Specifications

### Core Data Structures

```python
@dataclass
class AtomicOperation:
    name: str
    input_types: List[str]
    output_type: str
    compute: Callable
    preconditions: Optional[Callable]
    embedding: Optional[np.ndarray]
    success_contexts: List[dict]
```

### Problem Solving Pipeline

1. **Recognition Phase**: Extract patterns, identify candidate operations with confidence scores
2. **Fast Path**: Try high-confidence operations directly
3. **Slow Path**: Hierarchical beam search for compositions
4. **Learning Phase**: Update episodic memory, adjust operation weights, cache successful compositions

### Performance Targets

- Inference Time: <1 second for depth-5 search
- Learning Update: <100ms per episode
- Memory Growth: O(n) with episodes
- Accuracy: 90%+ on trained domains

## Critical Implementation Considerations

### Addressing Known Challenges

1. **Ontology Bottleneck**: Start with expert-defined operations (30-50 per domain), use composition to create virtual operations
2. **Cold Start Problem**: Use synthetic bootstrap, transfer learning, or hybrid LLM approach for initial suggestions
3. **Router Fragility**: Multi-strategy recognition with confidence thresholds and fallback mechanisms
4. **Combinatorial Explosion**: Hierarchical planning + beam search + learned heuristics + early termination

### Dependencies

```python
numpy>=1.20.0          # Numerical operations
scipy>=1.7.0           # Similarity computations
sentence-transformers  # Embedding generation
dataclasses           # Structure definitions
typing                # Type hints
```

### Minimum Resources

- CPU: Any modern processor
- RAM: 4GB minimum (16GB for production)
- Storage: 100MB for operation library + episodic memory
- GPU: Optional for embedding generation

## Architecture Documentation

### Core Architecture: `docs/CIG-Arch1.md`
The complete theoretical specification including:
- Detailed problem statement and design goals
- Component specifications with code examples
- System workflow diagrams
- Evaluation framework and benchmarks
- Risk assessment and mitigation strategies
- Example problem traces and operation library samples

### POC Implementation Plan: `docs/POC-updated.md`
Production-ready 7-week implementation guide including:
- **Week 0**: Problem generation and validation (with ground truth)
- **Week 1**: K_R operations, pattern recognition, and context extraction
- **Week 2**: K_P weight store and learning mechanism
- **Week 3**: Beam search with type checking and composite scoring
- **Week 4**: Episodic learning loop
- **Week 5**: Evaluation with confidence intervals (75-90% accuracy target)
- **Week 6**: Generalization testing on holdout set

**Critical Implementation Notes**:
- LCM operation uses correct formula: `(a * b) // gcd(a, b)`
- Type checking prevents invalid operation chains (e.g., `sum_list(12)`)
- POC supports single-chain problems only (multi-branch out of scope)
- Composite scoring offers both multiplication and weighted sum approaches
- Pattern recognition reduces search space by 5-10x
- Context extraction enables meaningful K_P learning (not all "generic")

**Success Criteria**:
- Core: 75-90% accuracy, 100% traceability, zero hallucinations
- Secondary: 30-50% depth reduction, interpretable traces
- Bonus: 85%+ training accuracy, 70%+ holdout accuracy

Refer to these documents for complete understanding of both theoretical foundation and practical implementation.
