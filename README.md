# ternary-scoring: Multi-criteria scoring and ranking for ternary strategies

Weighted linear scoring, Pareto front identification, min-max normalization, and leaderboard ranking for candidates evaluated on multiple criteria.

## Why This Exists

When you have several ternary strategies and multiple ways to evaluate them (speed, accuracy, cost, risk), you need to combine scores and rank candidates. A single weighted sum is one approach, but it hides trade-offs. This crate gives you both: a `WeightedScorer` for simple aggregation and a `ParetoScorer` for multi-objective analysis where no single candidate dominates all others.

## Core Concepts

- **Candidate** — A named entity with scores on named criteria. E.g., `{name: "strategy_a", scores: [("speed", 0.9), ("accuracy", 0.85)]}`.
- **ScoreFunction** — A trait for computing a single numeric score from a candidate. Implement this to define custom scoring logic.
- **WeightedScorer** — Computes Σ weight_i × score_i for specified criteria. Missing criteria default to 0. Supports custom weights or uniform weights.
- **ParetoScorer** — Identifies the Pareto front: the set of candidates where no other candidate is better on all objectives simultaneously. Each objective is marked maximize or minimize.
- **Pareto dominance** — Candidate A dominates candidate B if A is at least as good as B on every objective and strictly better on at least one.
- **ScoreNormalizer** — Min-max normalization: maps each criterion's scores to [0, 1] based on the observed min and max across all candidates.
- **Leaderboard** — Ranks candidates by a `ScoreFunction`, assigns ranks, and displays results.

## Quick Start

```toml
# Cargo.toml
[dependencies]
ternary-scoring = "0.1"
```

```rust
use ternary_scoring::*;

// Define candidates with multi-criteria scores
let candidates = vec![
    Candidate::new("conservative").with_score("speed", 0.4).with_score("accuracy", 0.95),
    Candidate::new("aggressive").with_score("speed", 0.9).with_score("accuracy", 0.7),
    Candidate::new("balanced").with_score("speed", 0.7).with_score("accuracy", 0.85),
];

// Weighted scoring: 60% speed, 40% accuracy
let scorer = WeightedScorer::new(vec![("speed", 0.6), ("accuracy", 0.4)]);
let lb = Leaderboard::from_scorer(&candidates, &scorer);
println!("{}", lb);
// #1 aggressive (0.8200)
// #2 balanced (0.7600)
// #3 conservative (0.6200)

// Pareto front: who is non-dominated?
let pareto = ParetoScorer::new(vec![("speed", true), ("accuracy", true)]);
let front = pareto.pareto_front(&candidates);
println!("Pareto front: {:?}", front); // [0, 1] — conservative and aggressive

// Normalize scores to [0, 1]
let normalizer = ScoreNormalizer::from_candidates(&candidates);
for c in &candidates {
    let normalized = normalizer.normalize(c);
    println!("{}: speed={:?}, accuracy={:?}",
        normalized.name,
        normalized.get_score("speed"),
        normalized.get_score("accuracy"));
}
```

## API Overview

| Type | What it is |
|---|---|
| `Candidate` | Named entity with multi-criteria scores |
| `ScoreFunction` | Trait: `name()` + `score(candidate) → f64` |
| `WeightedScorer` | Σ weight × score for named criteria |
| `ParetoScorer` | Identifies non-dominated candidates |
| `ScoreNormalizer` | Min-max normalization to [0, 1] |
| `Leaderboard` | Ranked list of candidates by score |
| `LeaderboardEntry` | One row: rank, name, score |

## How It Works

**Weighted scoring.** `WeightedScorer` stores a list of (criterion_name, weight) pairs. Scoring iterates this list, looks up each criterion in the candidate's scores (missing → 0.0), and sums weight × value. `uniform` distributes equal weight across all specified criteria.

**Pareto analysis.** `ParetoScorer::dominates(a, b)` checks every objective: for maximize objectives, a must be ≥ b; for minimize, a must be ≤ b. At least one must be strictly better. `pareto_front` runs O(n²) comparisons: for each candidate, it checks if any other candidate dominates it. Non-dominated candidates form the front.

**Normalization.** `ScoreNormalizer::from_candidates` scans all candidates to find the min and max of each criterion. `normalize` applies (value − min) / (max − min). If max = min (constant criterion), the normalized value is 0.0.

**Leaderboard.** `Leaderboard::from_scorer` scores all candidates, sorts descending by score, assigns ranks (1-indexed), and stores as `LeaderboardEntry` values. `winner()` returns the top entry. `rank_of(name)` looks up a specific candidate's rank.

## Known Limitations

- **Pareto front is O(n²).** Each candidate is compared against every other. For thousands of candidates, this gets slow. No approximation or fast non-dominated sorting (like NSGA-II) is implemented.
- **Single-value scores only.** Each criterion produces one `f64`. There's no support for score distributions, confidence intervals, or uncertainty ranges.
- **Leaderboard uses strict ordering.** Tied scores get different ranks based on sort stability. There's no tie-breaking or shared-rank logic.
- **`ParetoScorer::score` is a simple sum.** The `ScoreFunction` implementation for `ParetoScorer` just sums all objectives with equal weight. It's a convenience default, not a meaningful Pareto "score."

## Use Cases

- **Strategy selection.** Score candidate strategies on speed, accuracy, and resource usage. Use Pareto analysis to find the set of strategies where you'd need to make actual trade-offs.
- **Model comparison.** Rank ML models by weighted combination of precision, recall, and latency. Generate a leaderboard for reporting.
- **Benchmarking.** Normalize raw benchmark numbers (latency in ms, throughput in req/s, memory in MB) to [0, 1] so they're comparable, then score with custom weights.

## Ecosystem Context

Consumes candidates that may be produced by `ternary-pipeline` (pipeline output becomes scoring candidates) or `ternary-search` (search results as candidates). Output leaderboards and Pareto fronts feed into `ternary-scheduling` (top candidates get prioritized) and `ternary-validation` (validate top strategies against constraints).

## License

MIT
