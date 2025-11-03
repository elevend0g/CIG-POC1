#!/usr/bin/env python3
"""
Week 5: Evaluation Script - Comprehensive Metrics Analysis

Evaluates training results with detailed breakdowns by category,
problem type, and performance metrics.
"""

import sys
sys.path.insert(0, "/home/jay/ag0/cig/src")

import json
from collections import defaultdict


def load_episodes(filepath: str):
    """Load episodes from JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['episodes']


def categorize_problem(text: str) -> str:
    """Categorize problem by type"""
    text_lower = text.lower()

    if 'sum of all factors' in text_lower and 'even' not in text_lower:
        return 'factor_sum'
    elif 'product of all factors' in text_lower:
        return 'factor_product'
    elif 'how many factors' in text_lower:
        return 'factor_count'
    elif 'gcd' in text_lower:
        return 'gcd'
    elif 'lcm' in text_lower:
        return 'lcm'
    elif 'prime?' in text_lower or 'is' in text_lower and 'prime' in text_lower:
        return 'primality'
    elif 'sum of even factors' in text_lower:
        return 'even_factor_sum'
    elif 'largest factor' in text_lower and 'other than' in text_lower:
        return 'largest_proper_factor'
    else:
        return 'other'


def is_in_scope(category: str) -> bool:
    """Determine if problem category is in POC scope"""
    out_of_scope = {'gcd', 'lcm'}  # Binary operations not supported in single-chain POC
    return category not in out_of_scope


def evaluate_episodes(episodes):
    """Comprehensive evaluation of episodes"""

    # Overall metrics
    total = len(episodes)
    successes = sum(1 for ep in episodes if ep['goal_satisfied'])

    # By category
    by_category = defaultdict(lambda: {'total': 0, 'success': 0, 'depths': [], 'nodes': []})

    for ep in episodes:
        cat = categorize_problem(ep['problem_text'])
        by_category[cat]['total'] += 1
        if ep['goal_satisfied']:
            by_category[cat]['success'] += 1
        by_category[cat]['depths'].append(ep['search_depth'])
        by_category[cat]['nodes'].append(ep['nodes_explored'])

    # In-scope vs out-of-scope
    in_scope_total = 0
    in_scope_success = 0
    out_of_scope_total = 0
    out_of_scope_success = 0

    for cat, stats in by_category.items():
        if is_in_scope(cat):
            in_scope_total += stats['total']
            in_scope_success += stats['success']
        else:
            out_of_scope_total += stats['total']
            out_of_scope_success += stats['success']

    # Learning curve (10-problem windows)
    learning_curve = []
    window_size = 10
    for i in range(0, total, window_size):
        window = episodes[i:i+window_size]
        if window:
            window_success = sum(1 for ep in window if ep['goal_satisfied'])
            learning_curve.append({
                'window_start': i+1,
                'window_end': min(i+window_size, total),
                'success_rate': window_success / len(window),
                'count': len(window)
            })

    # Depth analysis
    successful_depths = [ep['search_depth'] for ep in episodes if ep['goal_satisfied']]
    failed_depths = [ep['search_depth'] for ep in episodes if not ep['goal_satisfied']]

    return {
        'overall': {
            'total': total,
            'successes': successes,
            'failures': total - successes,
            'success_rate': successes / total if total > 0 else 0,
            'avg_depth': sum(ep['search_depth'] for ep in episodes) / total if total > 0 else 0,
            'avg_nodes': sum(ep['nodes_explored'] for ep in episodes) / total if total > 0 else 0
        },
        'in_scope': {
            'total': in_scope_total,
            'successes': in_scope_success,
            'failures': in_scope_total - in_scope_success,
            'success_rate': in_scope_success / in_scope_total if in_scope_total > 0 else 0
        },
        'out_of_scope': {
            'total': out_of_scope_total,
            'successes': out_of_scope_success,
            'failures': out_of_scope_total - out_of_scope_success,
            'success_rate': out_of_scope_success / out_of_scope_total if out_of_scope_total > 0 else 0
        },
        'by_category': dict(by_category),
        'learning_curve': learning_curve,
        'depth_analysis': {
            'successful_avg': sum(successful_depths) / len(successful_depths) if successful_depths else 0,
            'failed_avg': sum(failed_depths) / len(failed_depths) if failed_depths else 0,
            'successful_depths': successful_depths,
            'failed_depths': failed_depths
        }
    }


def print_evaluation(results, title="Evaluation Results"):
    """Print formatted evaluation results"""

    print("=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Overall metrics
    overall = results['overall']
    print(f"\n📊 Overall Performance:")
    print(f"  Total problems:      {overall['total']}")
    print(f"  Successes:           {overall['successes']} ({overall['success_rate']*100:.1f}%)")
    print(f"  Failures:            {overall['failures']}")
    print(f"  Avg search depth:    {overall['avg_depth']:.2f} operations")
    print(f"  Avg nodes explored:  {overall['avg_nodes']:.1f}")

    # In-scope vs out-of-scope
    in_scope = results['in_scope']
    out_scope = results['out_of_scope']
    print(f"\n🎯 In-Scope Performance (Single-Chain Problems):")
    print(f"  Total:               {in_scope['total']} problems")
    print(f"  Successes:           {in_scope['successes']} ({in_scope['success_rate']*100:.1f}%)")
    print(f"  Failures:            {in_scope['failures']}")

    print(f"\n⚠️  Out-of-Scope Performance (Binary Operations - Not Supported):")
    print(f"  Total:               {out_scope['total']} problems")
    print(f"  Successes:           {out_scope['successes']} ({out_scope['success_rate']*100:.1f}%)")
    print(f"  Failures:            {out_scope['failures']}")

    # By category
    print(f"\n📋 Performance by Category:")
    by_cat = results['by_category']

    # Sort by success rate descending
    sorted_cats = sorted(by_cat.items(), key=lambda x: x[1]['success'] / x[1]['total'] if x[1]['total'] > 0 else 0, reverse=True)

    print(f"  {'Category':<25} {'Total':<8} {'Success':<10} {'Rate':<10} {'Avg Depth':<12} {'Scope'}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for cat, stats in sorted_cats:
        rate = stats['success'] / stats['total'] if stats['total'] > 0 else 0
        avg_depth = sum(stats['depths']) / len(stats['depths']) if stats['depths'] else 0
        scope = "✅ In" if is_in_scope(cat) else "❌ Out"
        print(f"  {cat:<25} {stats['total']:<8} {stats['success']:<10} {rate*100:<9.1f}% {avg_depth:<11.2f} {scope}")

    # Learning curve
    print(f"\n📈 Learning Curve (10-problem windows):")
    for window in results['learning_curve']:
        bar_length = int(window['success_rate'] * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"  Problems {window['window_start']:3d}-{window['window_end']:3d}: {bar} {window['success_rate']*100:5.1f}%")

    # Depth analysis
    depth = results['depth_analysis']
    print(f"\n🔍 Search Depth Analysis:")
    print(f"  Successful problems avg: {depth['successful_avg']:.2f} operations")
    print(f"  Failed problems avg:     {depth['failed_avg']:.2f} operations")

    # POC success criteria
    print(f"\n✅ POC Success Criteria Assessment:")
    in_scope_rate = in_scope['success_rate'] * 100

    if in_scope_rate >= 85:
        verdict = "🌟 EXCELLENT"
        desc = "POC exceeds expectations"
    elif in_scope_rate >= 75:
        verdict = "✅ TARGET MET"
        desc = "POC successfully demonstrates learning"
    elif in_scope_rate >= 65:
        verdict = "⚠️  ACCEPTABLE"
        desc = "Core concept proven, needs optimization"
    else:
        verdict = "❌ NEEDS WORK"
        desc = "Fundamental issues detected"

    print(f"  In-scope accuracy: {in_scope_rate:.1f}%")
    print(f"  Verdict: {verdict}")
    print(f"  Status: {desc}")


def save_evaluation(results, filepath: str):
    """Save evaluation results to JSON"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Evaluation saved to {filepath}")


if __name__ == "__main__":
    print("=" * 80)
    print("CIG Proof of Concept - Baseline Evaluation")
    print("=" * 80)

    # Load episodes
    print(f"\n[1/3] Loading episodes...")
    episodes = load_episodes("data/results/episodes_all.json")
    print(f"✅ Loaded {len(episodes)} episodes")

    # Evaluate
    print(f"\n[2/3] Computing metrics...")
    results = evaluate_episodes(episodes)
    print(f"✅ Analysis complete")

    # Print results
    print(f"\n[3/3] Results:")
    print_evaluation(results, title="BASELINE EVALUATION (Before Fix)")

    # Save
    save_evaluation(results, "data/results/evaluation_baseline.json")

    print("\n" + "=" * 80)
    print("Baseline evaluation complete!")
    print("=" * 80)
    print("\nNext: Fix max_excluding_self pattern recognition")
