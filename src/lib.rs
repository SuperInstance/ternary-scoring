//! Multi-criteria scoring and ranking for ternary strategies.
//!
//! Combine multiple numeric criteria (speed, accuracy, cost, risk, ...) into
//! composite scores, identify Pareto-optimal candidates, normalize raw values
//! to a comparable range, and produce ranked leaderboards.
//!
//! # When to use this crate
//!
//! Use `ternary-scoring` when you have a small-to-medium set of candidates and
//! several numerical criteria per candidate, and you want to:
//!
//! - aggregate criteria into a single weighted score ([`WeightedScorer`]),
//! - find the set of candidates where no other is better on every objective
//!   ([`ParetoScorer`]),
//! - rescale heterogeneous units (ms, MB, req/s) onto a common `[0, 1]` range
//!   ([`ScoreNormalizer`]),
//! - rank candidates and pick a winner ([`Leaderboard`]).
//!
//! # Quick example
//!
//! ```
//! use ternary_scoring::*;
//!
//! let candidates = vec![
//!     Candidate::new("conservative")
//!         .with_score("speed", 0.4)
//!         .with_score("accuracy", 0.95),
//!     Candidate::new("aggressive")
//!         .with_score("speed", 0.9)
//!         .with_score("accuracy", 0.7),
//!     Candidate::new("balanced")
//!         .with_score("speed", 0.7)
//!         .with_score("accuracy", 0.85),
//! ];
//!
//! // 60% speed, 40% accuracy
//! let scorer = WeightedScorer::new(vec![("speed", 0.6), ("accuracy", 0.4)]);
//! let lb = Leaderboard::from_scorer(&candidates, &scorer);
//! assert_eq!(lb.winner().unwrap().name, "aggressive");
//! assert!((lb.winner().unwrap().score - 0.82).abs() < 1e-9);
//! ```

use core::fmt;

/// A named candidate carrying one or more named numerical criterion scores.
///
/// Each entry in [`Candidate::scores`] is a `(criterion_name, value)` pair.
/// Criteria are looked up by name via [`Candidate::get_score`], which returns
/// the *first* matching entry. Builders like [`Candidate::with_score`] append,
/// so adding the same criterion twice stores two entries (only the first is
/// seen by `get_score`).
#[derive(Debug, Clone)]
pub struct Candidate {
    pub name: String,
    pub scores: Vec<(String, f64)>,
}

impl Candidate {
    /// Create a new candidate with no scores yet.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            scores: Vec::new(),
        }
    }

    /// Append a `(criterion, value)` score to this candidate (builder-style).
    ///
    /// This does *not* overwrite an existing entry for the same criterion;
    /// [`Candidate::get_score`] returns the first match.
    pub fn with_score(mut self, criterion: &str, value: f64) -> Self {
        self.scores.push((criterion.to_string(), value));
        self
    }

    /// Look up the first score recorded for `criterion`, or `None` if absent.
    pub fn get_score(&self, criterion: &str) -> Option<f64> {
        self.scores
            .iter()
            .find(|(k, _)| k == criterion)
            .map(|(_, v)| *v)
    }
}

/// A function that maps a [`Candidate`] to a single `f64` score.
///
/// Implement this to plug a custom aggregation strategy into [`Leaderboard`].
pub trait ScoreFunction {
    /// Human-readable label for this scoring function (used for diagnostics).
    fn name(&self) -> &str;
    /// Compute the composite score for `candidate`. Higher is better.
    fn score(&self, candidate: &Candidate) -> f64;
}

/// Weighted linear scorer: `score = Σ weight_i × value_i`.
///
/// Criteria not present on a candidate contribute `0.0`. Build with
/// [`WeightedScorer::new`] for explicit weights or [`WeightedScorer::uniform`]
/// for equal `1/n` weights across the listed criteria.
pub struct WeightedScorer {
    /// Ordered list of `(criterion, weight)` pairs applied during scoring.
    pub weights: Vec<(String, f64)>,
}

impl WeightedScorer {
    /// Construct from explicit `(criterion, weight)` pairs.
    pub fn new(weights: Vec<(&str, f64)>) -> Self {
        Self {
            weights: weights
                .into_iter()
                .map(|(k, w)| (k.to_string(), w))
                .collect(),
        }
    }

    /// Construct with equal weight `1/n` for each of the `n` keys.
    ///
    /// Empty input yields an empty scorer (no division is performed, so no
    /// `inf` weight is ever produced).
    pub fn uniform(keys: &[&str]) -> Self {
        if keys.is_empty() {
            return Self::new(Vec::new());
        }
        let w = 1.0 / keys.len() as f64;
        Self::new(keys.iter().map(|k| (*k, w)).collect())
    }
}

impl ScoreFunction for WeightedScorer {
    fn name(&self) -> &str {
        "weighted"
    }
    fn score(&self, candidate: &Candidate) -> f64 {
        self.weights
            .iter()
            .map(|(k, w)| candidate.get_score(k).unwrap_or(0.0) * w)
            .sum()
    }
}

/// Multi-objective Pareto dominance analyzer.
///
/// Each objective is named and flagged `maximize = true` (higher is better) or
/// `maximize = false` (lower is better). See [`ParetoScorer::dominates`] for
/// the dominance definition and [`ParetoScorer::pareto_front`] for the
/// non-dominated subset.
#[derive(Debug, Clone)]
pub struct ParetoScorer {
    /// Objective criterion names, in declaration order.
    pub objectives: Vec<String>,
    /// Parallel to [`ParetoScorer::objectives`]: `true` means maximize,
    /// `false` means minimize.
    pub maximize: Vec<bool>,
}

impl ParetoScorer {
    /// Construct from `(criterion, maximize)` pairs.
    pub fn new(objectives: Vec<(&str, bool)>) -> Self {
        Self {
            objectives: objectives.iter().map(|(k, _)| k.to_string()).collect(),
            maximize: objectives.iter().map(|(_, m)| *m).collect(),
        }
    }

    /// Standard Pareto dominance: `a` dominates `b` iff `a` is at least as good
    /// as `b` on every objective and strictly better on at least one.
    ///
    /// Objective values missing from a candidate are treated as `0.0`.
    /// Comparisons involving `NaN` always fail, so a candidate with a `NaN`
    /// objective can neither dominate nor be dominated through that objective.
    pub fn dominates(&self, a: &Candidate, b: &Candidate) -> bool {
        let mut at_least_one_better = false;
        for (i, obj) in self.objectives.iter().enumerate() {
            let av = a.get_score(obj).unwrap_or(0.0);
            let bv = b.get_score(obj).unwrap_or(0.0);
            let (better, at_least_as_good) = if self.maximize[i] {
                (av > bv, av >= bv)
            } else {
                (av < bv, av <= bv)
            };
            if better {
                at_least_one_better = true;
            }
            if !at_least_as_good {
                return false;
            }
        }
        at_least_one_better
    }

    /// Returns the indices of all non-dominated candidates (the Pareto front).
    ///
    /// Performs O(n²) dominance comparisons. The order of returned indices
    /// follows the input order.
    pub fn pareto_front(&self, candidates: &[Candidate]) -> Vec<usize> {
        let mut front = Vec::new();
        for i in 0..candidates.len() {
            let dominated = candidates
                .iter()
                .enumerate()
                .any(|(j, b)| j != i && self.dominates(b, &candidates[i]));
            if !dominated {
                front.push(i);
            }
        }
        front
    }
}

impl ScoreFunction for ParetoScorer {
    fn name(&self) -> &str {
        "pareto"
    }
    fn score(&self, candidate: &Candidate) -> f64 {
        // Convenience default: unweighted sum of objective values. This is not
        // a meaningful Pareto "score" (Pareto analysis is ordinal); it exists
        // only so `ParetoScorer` can satisfy `ScoreFunction` for use with
        // `Leaderboard` when a numeric proxy is needed.
        self.objectives
            .iter()
            .map(|k| candidate.get_score(k).unwrap_or(0.0))
            .sum()
    }
}

/// Min-max normalizer that rescales each criterion's observed values to
/// `[0, 1]` based on the minimum and maximum seen across the training
/// candidates.
///
/// Build with [`ScoreNormalizer::from_candidates`], then call
/// [`ScoreNormalizer::normalize`] per candidate.
pub struct ScoreNormalizer {
    /// Minimum observed value per criterion (used as the lower anchor).
    pub mins: Vec<(String, f64)>,
    /// Maximum observed value per criterion (used as the upper anchor).
    pub maxs: Vec<(String, f64)>,
}

impl ScoreNormalizer {
    /// Build a normalizer by scanning every candidate for the union of all
    /// criterion names, then computing per-criterion min and max across all
    /// candidates that define that criterion.
    ///
    /// Criteria absent from all candidates are not tracked. If a candidate
    /// later normalized by this `ScoreNormalizer` references an untracked
    /// criterion, that value is passed through unchanged.
    pub fn from_candidates(candidates: &[Candidate]) -> Self {
        // Collect the union of all criteria across every candidate, preserving
        // first-seen order. (Earlier versions only inspected the first
        // candidate, which silently dropped criteria unique to others.)
        let mut criteria: Vec<String> = Vec::new();
        for c in candidates {
            for (k, _) in &c.scores {
                if !criteria.iter().any(|existing| existing == k) {
                    criteria.push(k.clone());
                }
            }
        }
        let mins = criteria
            .iter()
            .map(|k| {
                let min = candidates
                    .iter()
                    .filter_map(|c| c.get_score(k))
                    .reduce(f64::min)
                    .unwrap_or(0.0);
                (k.clone(), min)
            })
            .collect();
        let maxs = criteria
            .iter()
            .map(|k| {
                let max = candidates
                    .iter()
                    .filter_map(|c| c.get_score(k))
                    .reduce(f64::max)
                    .unwrap_or(1.0);
                (k.clone(), max)
            })
            .collect();
        Self { mins, maxs }
    }

    /// Return a new candidate whose scores are `(value - min) / (max - min)`
    /// for each tracked criterion. A criterion whose observed range is zero
    /// (`max == min`) is mapped to `0.0`. Criteria absent from this
    /// normalizer's training set are passed through unchanged.
    pub fn normalize(&self, candidate: &Candidate) -> Candidate {
        let mut result = Candidate::new(&candidate.name);
        for (k, v) in &candidate.scores {
            let min = lookup(self.mins.as_slice(), k);
            let max = lookup(self.maxs.as_slice(), k);
            let norm = match (min, max) {
                (Some(min), Some(max)) if (max - min).abs() >= f64::EPSILON => {
                    (v - min) / (max - min)
                }
                // Constant or tracked range: collapse to 0.0 per spec.
                (Some(_), Some(_)) => 0.0,
                // Untracked criterion: pass through unchanged.
                _ => *v,
            };
            result = result.with_score(k, norm);
        }
        result
    }
}

/// Look up the numeric value associated with `key` in a `(String, f64)` list.
fn lookup(pairs: &[(String, f64)], key: &str) -> Option<f64> {
    pairs.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
}

/// One ranked row in a [`Leaderboard`].
#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    /// 1-indexed rank within the leaderboard (1 = best).
    pub rank: usize,
    /// Candidate name.
    pub name: String,
    /// Composite score produced by the [`ScoreFunction`].
    pub score: f64,
}

/// A ranked view of candidates produced by sorting descending by a
/// [`ScoreFunction`].
///
/// Ties are broken by sort stability only — there is no shared-rank logic.
pub struct Leaderboard {
    /// Entries sorted best-first.
    pub entries: Vec<LeaderboardEntry>,
}

impl Leaderboard {
    /// Score every candidate, sort descending by score, and assign 1-indexed
    /// ranks.
    ///
    /// Ordering uses [`f64::total_cmp`] so the sort is total and panic-free.
    /// `NaN` scores are explicitly treated as *worse* than any real value
    /// (including `-infinity`), so a candidate whose score is `NaN` always
    /// sorts to the bottom of the leaderboard rather than silently "winning".
    pub fn from_scorer(candidates: &[Candidate], scorer: &dyn ScoreFunction) -> Self {
        let mut scored: Vec<(String, f64)> = candidates
            .iter()
            .map(|c| (c.name.clone(), scorer.score(c)))
            .collect();
        // Descending sort: higher scores first, NaNs last.
        scored.sort_by(|a, b| compare_scores_desc(a.1, b.1));
        let entries = scored
            .into_iter()
            .enumerate()
            .map(|(i, (name, score))| LeaderboardEntry {
                rank: i + 1,
                name,
                score,
            })
            .collect();
        Self { entries }
    }

    /// The top-ranked entry, or `None` if the leaderboard is empty.
    pub fn winner(&self) -> Option<&LeaderboardEntry> {
        self.entries.first()
    }

    /// The 1-indexed rank of the candidate named `name`, or `None` if not
    /// present. If multiple candidates share a name, the first match wins.
    pub fn rank_of(&self, name: &str) -> Option<usize> {
        self.entries.iter().find(|e| e.name == name).map(|e| e.rank)
    }
}

impl fmt::Display for Leaderboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in &self.entries {
            writeln!(f, "#{} {} ({:.4})", e.rank, e.name, e.score)?;
        }
        Ok(())
    }
}

/// Ordering helper for descending leaderboard sort.
///
/// Given two scores `a` and `b`, returns the [`Ordering`] of `a` relative to
/// `b` such that *higher is better* and `NaN` is treated as worse than every
/// real value (including `-infinity`). Suitable for passing directly to
/// [`slice::sort_by`].
fn compare_scores_desc(a: f64, b: f64) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        // `a` is NaN: NaN is worst, so `a` sorts after `b`.
        (true, false) => Ordering::Greater,
        // `b` is NaN: `a` (real) is better, so `a` sorts before `b`.
        (false, true) => Ordering::Less,
        // Both real: descending by total order.
        (false, false) => b.total_cmp(&a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(name: &str, speed: f64, accuracy: f64) -> Candidate {
        Candidate::new(name)
            .with_score("speed", speed)
            .with_score("accuracy", accuracy)
    }

    // ---------- Candidate ----------

    #[test]
    fn test_candidate_get_score() {
        let c = make_candidate("a", 0.9, 0.8);
        assert_eq!(c.get_score("speed"), Some(0.9));
        assert_eq!(c.get_score("accuracy"), Some(0.8));
        assert_eq!(c.get_score("other"), None);
    }

    #[test]
    fn test_candidate_with_multiple_scores() {
        let c = Candidate::new("multi")
            .with_score("a", 1.0)
            .with_score("b", 2.0)
            .with_score("c", 3.0);
        assert_eq!(c.scores.len(), 3);
    }

    #[test]
    fn test_candidate_duplicate_criterion_returns_first() {
        // with_score appends; get_score returns the first match.
        let c = Candidate::new("x")
            .with_score("a", 1.0)
            .with_score("a", 2.0);
        assert_eq!(c.get_score("a"), Some(1.0));
        assert_eq!(c.scores.len(), 2);
    }

    // ---------- WeightedScorer ----------

    #[test]
    fn test_weighted_scorer() {
        // Hand calculation: 0.6 * 1.0 + 0.4 * 0.5 = 0.6 + 0.2 = 0.8
        let scorer = WeightedScorer::new(vec![("speed", 0.6), ("accuracy", 0.4)]);
        let c = make_candidate("a", 1.0, 0.5);
        let score = scorer.score(&c);
        assert!((score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_uniform() {
        // 1/2 * 1.0 + 1/2 * 1.0 = 1.0
        let scorer = WeightedScorer::uniform(&["speed", "accuracy"]);
        let c = make_candidate("a", 1.0, 1.0);
        let score = scorer.score(&c);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_uniform_mid_value() {
        // Independent check: 1/2 * 0.4 + 1/2 * 0.8 = 0.6
        let scorer = WeightedScorer::uniform(&["speed", "accuracy"]);
        let c = make_candidate("a", 0.4, 0.8);
        assert!((scorer.score(&c) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_uniform_empty_is_safe() {
        // No division-by-zero panic; produces an empty scorer that scores 0.
        let scorer = WeightedScorer::uniform(&[]);
        assert!(scorer.weights.is_empty());
        let c = Candidate::new("x").with_score("anything", 42.0);
        assert_eq!(scorer.score(&c), 0.0);
    }

    #[test]
    fn test_weighted_missing_criterion() {
        let scorer = WeightedScorer::new(vec![("speed", 1.0)]);
        let c = Candidate::new("x");
        assert_eq!(scorer.score(&c), 0.0);
    }

    #[test]
    fn test_weighted_all_zero_weights() {
        let scorer = WeightedScorer::new(vec![("speed", 0.0), ("accuracy", 0.0)]);
        let c = make_candidate("a", 100.0, 100.0);
        assert_eq!(scorer.score(&c), 0.0);
    }

    #[test]
    fn test_scorer_name() {
        let w = WeightedScorer::new(vec![]);
        assert_eq!(w.name(), "weighted");
        let p = ParetoScorer::new(vec![]);
        assert_eq!(p.name(), "pareto");
    }

    // ---------- ParetoScorer ----------

    #[test]
    fn test_pareto_dominates() {
        let p = ParetoScorer::new(vec![("speed", true), ("cost", false)]);
        let a = Candidate::new("a")
            .with_score("speed", 10.0)
            .with_score("cost", 5.0);
        let b = Candidate::new("b")
            .with_score("speed", 8.0)
            .with_score("cost", 6.0);
        // a: faster (10>8, maximize) and cheaper (5<6, minimize) -> dominates
        assert!(p.dominates(&a, &b));
        assert!(!p.dominates(&b, &a));
    }

    #[test]
    fn test_pareto_no_dominance() {
        let p = ParetoScorer::new(vec![("speed", true), ("cost", false)]);
        let a = Candidate::new("a")
            .with_score("speed", 10.0)
            .with_score("cost", 10.0);
        let b = Candidate::new("b")
            .with_score("speed", 5.0)
            .with_score("cost", 5.0);
        assert!(!p.dominates(&a, &b)); // a has better speed but worse cost
        assert!(!p.dominates(&b, &a));
    }

    #[test]
    fn test_pareto_front() {
        // speed maximize, cost minimize.
        // a (10, 5) dominates c (5, 10): faster AND cheaper.
        // b (8, 3) also dominates c: faster AND cheaper.
        // c is dominated; front = {a, b}.
        let p = ParetoScorer::new(vec![("speed", true), ("cost", false)]);
        let a = Candidate::new("a")
            .with_score("speed", 10.0)
            .with_score("cost", 5.0);
        let b = Candidate::new("b")
            .with_score("speed", 8.0)
            .with_score("cost", 3.0);
        let c = Candidate::new("c")
            .with_score("speed", 5.0)
            .with_score("cost", 10.0);
        let front = p.pareto_front(&[a, b, c]);
        assert_eq!(front.len(), 2);
        assert!(front.contains(&0)); // a
        assert!(front.contains(&1)); // b
    }

    #[test]
    fn test_pareto_front_readme_scenario() {
        // Reproduces the README Quick Start scenario exactly. None of the
        // three candidates dominates another (each trades speed for
        // accuracy), so the front is the full set [0, 1, 2].
        let candidates = vec![
            Candidate::new("conservative")
                .with_score("speed", 0.4)
                .with_score("accuracy", 0.95),
            Candidate::new("aggressive")
                .with_score("speed", 0.9)
                .with_score("accuracy", 0.7),
            Candidate::new("balanced")
                .with_score("speed", 0.7)
                .with_score("accuracy", 0.85),
        ];
        let p = ParetoScorer::new(vec![("speed", true), ("accuracy", true)]);
        let front = p.pareto_front(&candidates);
        assert_eq!(front, vec![0, 1, 2]);
    }

    #[test]
    fn test_pareto_score() {
        let p = ParetoScorer::new(vec![("a", true), ("b", true)]);
        let c = Candidate::new("x")
            .with_score("a", 3.0)
            .with_score("b", 4.0);
        assert_eq!(p.score(&c), 7.0);
    }

    #[test]
    fn test_pareto_single_candidate() {
        let p = ParetoScorer::new(vec![("speed", true)]);
        let a = Candidate::new("a").with_score("speed", 5.0);
        let front = p.pareto_front(&[a]);
        assert_eq!(front, vec![0]);
    }

    #[test]
    fn test_pareto_identical_candidates() {
        // Identical candidates do not dominate each other (no strict
        // improvement on any objective).
        let p = ParetoScorer::new(vec![("x", true)]);
        let a = Candidate::new("a").with_score("x", 1.0);
        let b = Candidate::new("b").with_score("x", 1.0);
        let front = p.pareto_front(&[a, b]);
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn test_pareto_empty() {
        let p = ParetoScorer::new(vec![("x", true)]);
        let front = p.pareto_front(&[]);
        assert!(front.is_empty());
    }

    #[test]
    fn test_pareto_nan_does_not_dominate() {
        // NaN comparisons always fail, so NaN candidates neither dominate nor
        // are dominated through the NaN objective.
        let p = ParetoScorer::new(vec![("x", true)]);
        let a = Candidate::new("a").with_score("x", f64::NAN);
        let b = Candidate::new("b").with_score("x", 1.0);
        assert!(!p.dominates(&a, &b));
        assert!(!p.dominates(&b, &a));
    }

    // ---------- ScoreNormalizer ----------

    #[test]
    fn test_normalizer() {
        let c1 = make_candidate("a", 10.0, 0.5);
        let c2 = make_candidate("b", 20.0, 1.0);
        let norm = ScoreNormalizer::from_candidates(&[c1.clone(), c2.clone()]);
        let n1 = norm.normalize(&c1);
        assert_eq!(n1.get_score("speed"), Some(0.0));
        assert_eq!(n1.get_score("accuracy"), Some(0.0));
        let n2 = norm.normalize(&c2);
        assert_eq!(n2.get_score("speed"), Some(1.0));
        assert_eq!(n2.get_score("accuracy"), Some(1.0));
    }

    #[test]
    fn test_normalizer_mid_value_hand_calculation() {
        // Independent hand calculation, not just boundary 0/1 values.
        // speed range [10, 20]; a value of 15 -> (15-10)/(20-10) = 0.5
        // accuracy range [0.5, 1.0]; a value of 0.75 -> (0.75-0.5)/(1.0-0.5) = 0.5
        let c1 = make_candidate("a", 10.0, 0.5);
        let c2 = make_candidate("b", 20.0, 1.0);
        let mid = make_candidate("m", 15.0, 0.75);
        let norm = ScoreNormalizer::from_candidates(&[c1, c2]);
        let n = norm.normalize(&mid);
        assert!((n.get_score("speed").unwrap() - 0.5).abs() < 1e-9);
        assert!((n.get_score("accuracy").unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_normalizer_constant() {
        let c1 = Candidate::new("a").with_score("x", 5.0);
        let c2 = Candidate::new("b").with_score("x", 5.0);
        let norm = ScoreNormalizer::from_candidates(&[c1.clone(), c2.clone()]);
        let n = norm.normalize(&c1);
        assert_eq!(n.get_score("x"), Some(0.0)); // constant -> 0
    }

    #[test]
    fn test_normalizer_empty_candidates() {
        // No candidates => no tracked criteria; normalize passes through.
        let norm = ScoreNormalizer::from_candidates(&[]);
        let c = Candidate::new("a").with_score("x", 7.0);
        let n = norm.normalize(&c);
        assert_eq!(n.get_score("x"), Some(7.0));
    }

    #[test]
    fn test_normalizer_untracked_criterion_passes_through() {
        // A criterion absent from training candidates is passed through
        // unchanged rather than silently rescaled.
        let c1 = Candidate::new("a").with_score("x", 1.0);
        let c2 = Candidate::new("b").with_score("x", 3.0);
        let norm = ScoreNormalizer::from_candidates(&[c1, c2]);
        let c_with_new = Candidate::new("m").with_score("y", 42.0);
        let n = norm.normalize(&c_with_new);
        assert_eq!(n.get_score("y"), Some(42.0));
    }

    #[test]
    fn test_normalizer_unions_criteria_across_candidates() {
        // Regression: previously only the first candidate's criteria were
        // tracked, so 'cost' (only on candidate 2) was silently dropped.
        // Now criteria are unioned across all candidates.
        let c1 = Candidate::new("a")
            .with_score("speed", 10.0)
            .with_score("accuracy", 0.5);
        let c2 = Candidate::new("b")
            .with_score("speed", 20.0)
            .with_score("accuracy", 1.0)
            .with_score("cost", 100.0);
        let c3 = Candidate::new("c")
            .with_score("speed", 15.0)
            .with_score("accuracy", 0.75)
            .with_score("cost", 200.0);
        let norm = ScoreNormalizer::from_candidates(&[c1, c2, c3.clone()]);
        // cost range is [100, 200]; c3.cost=200 -> 1.0
        let n3 = norm.normalize(&c3);
        assert!((n3.get_score("cost").unwrap() - 1.0).abs() < 1e-9);
        // c2.cost=100 -> 0.0
        let c2_again = Candidate::new("b")
            .with_score("speed", 20.0)
            .with_score("accuracy", 1.0)
            .with_score("cost", 100.0);
        let n2 = norm.normalize(&c2_again);
        assert!((n2.get_score("cost").unwrap() - 0.0).abs() < 1e-9);
    }

    // ---------- Leaderboard ----------

    #[test]
    fn test_leaderboard_ranking() {
        let scorer = WeightedScorer::new(vec![("speed", 1.0)]);
        let c1 = Candidate::new("slow").with_score("speed", 1.0);
        let c2 = Candidate::new("fast").with_score("speed", 10.0);
        let lb = Leaderboard::from_scorer(&[c1, c2], &scorer);
        assert_eq!(lb.winner().unwrap().name, "fast");
        assert_eq!(lb.rank_of("fast"), Some(1));
        assert_eq!(lb.rank_of("slow"), Some(2));
    }

    #[test]
    fn test_leaderboard_readme_quickstart_hand_calculation() {
        // Independent hand verification of the README's headline numbers.
        //   aggressive : 0.6*0.9 + 0.4*0.7 = 0.54 + 0.28 = 0.82
        //   balanced   : 0.6*0.7 + 0.4*0.85 = 0.42 + 0.34 = 0.76
        //   conservative: 0.6*0.4 + 0.4*0.95 = 0.24 + 0.38 = 0.62
        let candidates = vec![
            Candidate::new("conservative")
                .with_score("speed", 0.4)
                .with_score("accuracy", 0.95),
            Candidate::new("aggressive")
                .with_score("speed", 0.9)
                .with_score("accuracy", 0.7),
            Candidate::new("balanced")
                .with_score("speed", 0.7)
                .with_score("accuracy", 0.85),
        ];
        let scorer = WeightedScorer::new(vec![("speed", 0.6), ("accuracy", 0.4)]);
        let lb = Leaderboard::from_scorer(&candidates, &scorer);
        assert_eq!(lb.entries.len(), 3);
        assert_eq!(lb.entries[0].name, "aggressive");
        assert_eq!(lb.entries[1].name, "balanced");
        assert_eq!(lb.entries[2].name, "conservative");
        assert!((lb.entries[0].score - 0.82).abs() < 1e-9);
        assert!((lb.entries[1].score - 0.76).abs() < 1e-9);
        assert!((lb.entries[2].score - 0.62).abs() < 1e-9);
        // Verify the Display output matches the README format exactly.
        let rendered = format!("{}", lb);
        assert_eq!(
            rendered,
            "#1 aggressive (0.8200)\n#2 balanced (0.7600)\n#3 conservative (0.6200)\n"
        );
    }

    #[test]
    fn test_leaderboard_display() {
        let scorer = WeightedScorer::new(vec![("speed", 1.0)]);
        let c = Candidate::new("x").with_score("speed", 5.0);
        let lb = Leaderboard::from_scorer(&[c], &scorer);
        let s = format!("{}", lb);
        assert!(s.contains("x"));
    }

    #[test]
    fn test_leaderboard_empty() {
        let scorer = WeightedScorer::new(vec![]);
        let lb = Leaderboard::from_scorer(&[], &scorer);
        assert!(lb.winner().is_none());
    }

    #[test]
    fn test_leaderboard_rank_of_missing() {
        let scorer = WeightedScorer::new(vec![("speed", 1.0)]);
        let c = Candidate::new("a").with_score("speed", 1.0);
        let lb = Leaderboard::from_scorer(&[c], &scorer);
        assert_eq!(lb.rank_of("nonexistent"), None);
    }

    #[test]
    fn test_leaderboard_nan_score_does_not_panic() {
        // total_cmp must keep this panic-free; NaN sorts as smallest, so the
        // NaN candidate ranks last.
        let scorer = WeightedScorer::new(vec![("x", 1.0)]);
        let good = Candidate::new("good").with_score("x", 1.0);
        let bad = Candidate::new("bad").with_score("x", f64::NAN);
        let lb = Leaderboard::from_scorer(&[good, bad], &scorer);
        assert_eq!(lb.entries[0].name, "good");
        assert_eq!(lb.entries[1].name, "bad");
        assert!(lb.entries[1].score.is_nan());
    }

    #[test]
    fn test_three_way_leaderboard() {
        let scorer = WeightedScorer::new(vec![("a", 0.5), ("b", 0.5)]);
        let c1 = Candidate::new("low")
            .with_score("a", 0.0)
            .with_score("b", 0.0);
        let c2 = Candidate::new("mid")
            .with_score("a", 0.5)
            .with_score("b", 0.5);
        let c3 = Candidate::new("high")
            .with_score("a", 1.0)
            .with_score("b", 1.0);
        let lb = Leaderboard::from_scorer(&[c1, c2, c3], &scorer);
        assert_eq!(lb.entries[0].name, "high");
        assert_eq!(lb.entries[2].name, "low");
    }
}
