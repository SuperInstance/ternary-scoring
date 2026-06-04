//! Multi-criteria scoring for ternary strategies.
//!
//! Provides `ScoreFunction`, `WeightedScorer`, `ParetoScorer`, `ScoreNormalizer`,
//! and `Leaderboard` for ranking ternary strategies.

use core::fmt;

/// A named candidate to be scored.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub name: String,
    pub scores: Vec<(String, f64)>,
}

impl Candidate {
    pub fn new(name: &str) -> Self {
        Self { name: name.to_string(), scores: Vec::new() }
    }

    pub fn with_score(mut self, criterion: &str, value: f64) -> Self {
        self.scores.push((criterion.to_string(), value));
        self
    }

    pub fn get_score(&self, criterion: &str) -> Option<f64> {
        self.scores.iter().find(|(k, _)| k == criterion).map(|(_, v)| *v)
    }
}

/// A scoring function trait.
pub trait ScoreFunction {
    fn name(&self) -> &str;
    fn score(&self, candidate: &Candidate) -> f64;
}

/// A simple weighted linear scorer.
pub struct WeightedScorer {
    pub weights: Vec<(String, f64)>,
}

impl WeightedScorer {
    pub fn new(weights: Vec<(&str, f64)>) -> Self {
        Self {
            weights: weights.into_iter().map(|(k, w)| (k.to_string(), w)).collect(),
        }
    }

    pub fn uniform(keys: &[&str]) -> Self {
        let w = 1.0 / keys.len() as f64;
        Self::new(keys.iter().map(|k| (*k, w)).collect())
    }
}

impl ScoreFunction for WeightedScorer {
    fn name(&self) -> &str { "weighted" }
    fn score(&self, candidate: &Candidate) -> f64 {
        self.weights.iter().map(|(k, w)| {
            candidate.get_score(k).unwrap_or(0.0) * w
        }).sum()
    }
}

/// A Pareto (non-dominated) scorer for multi-objective optimization.
#[derive(Debug, Clone)]
pub struct ParetoScorer {
    pub objectives: Vec<String>,
    /// true = maximize, false = minimize
    pub maximize: Vec<bool>,
}

impl ParetoScorer {
    pub fn new(objectives: Vec<(&str, bool)>) -> Self {
        Self {
            objectives: objectives.iter().map(|(k, _)| k.to_string()).collect(),
            maximize: objectives.iter().map(|(_, m)| *m).collect(),
        }
    }

    /// Returns true if `a` dominates `b`.
    pub fn dominates(&self, a: &Candidate, b: &Candidate) -> bool {
        let mut at_least_one_better = false;
        for (i, obj) in self.objectives.iter().enumerate() {
            let av = a.get_score(obj).unwrap_or(0.0);
            let bv = b.get_score(obj).unwrap_or(0.0);
            let (better, equal) = if self.maximize[i] {
                (av > bv, av >= bv)
            } else {
                (av < bv, av <= bv)
            };
            if better { at_least_one_better = true; }
            if !equal { return false; }
        }
        at_least_one_better
    }

    /// Returns indices of the Pareto front (non-dominated candidates).
    pub fn pareto_front(&self, candidates: &[Candidate]) -> Vec<usize> {
        let mut front = Vec::new();
        for i in 0..candidates.len() {
            let dominated = candidates.iter().enumerate().any(|(j, b)| {
                j != i && self.dominates(b, &candidates[i])
            });
            if !dominated {
                front.push(i);
            }
        }
        front
    }
}

impl ScoreFunction for ParetoScorer {
    fn name(&self) -> &str { "pareto" }
    fn score(&self, candidate: &Candidate) -> f64 {
        // For Pareto, score is just a simple weighted sum of objectives (equal weight)
        self.objectives.iter().map(|k| candidate.get_score(k).unwrap_or(0.0)).sum()
    }
}

/// Normalizes scores to [0, 1] range using min-max normalization.
pub struct ScoreNormalizer {
    pub mins: Vec<(String, f64)>,
    pub maxs: Vec<(String, f64)>,
}

impl ScoreNormalizer {
    /// Compute normalizer from a set of candidates.
    pub fn from_candidates(candidates: &[Candidate]) -> Self {
        let mut criteria: Vec<String> = Vec::new();
        if let Some(c) = candidates.first() {
            for (k, _) in &c.scores {
                criteria.push(k.clone());
            }
        }
        let mins = criteria.iter().map(|k| {
            let min = candidates.iter()
                .filter_map(|c| c.get_score(k))
                .reduce(f64::min)
                .unwrap_or(0.0);
            (k.clone(), min)
        }).collect();
        let maxs = criteria.iter().map(|k| {
            let max = candidates.iter()
                .filter_map(|c| c.get_score(k))
                .reduce(f64::max)
                .unwrap_or(1.0);
            (k.clone(), max)
        }).collect();
        Self { mins, maxs }
    }

    pub fn normalize(&self, candidate: &Candidate) -> Candidate {
        let mut result = Candidate::new(&candidate.name);
        for (k, v) in &candidate.scores {
            let min = self.mins.iter().find(|(key, _)| key == k).map(|(_, v)| *v).unwrap_or(0.0);
            let max = self.maxs.iter().find(|(key, _)| key == k).map(|(_, v)| *v).unwrap_or(1.0);
            let norm = if (max - min).abs() < f64::EPSILON { 0.0 } else { (v - min) / (max - min) };
            result = result.with_score(k, norm);
        }
        result
    }
}

/// Entry in a leaderboard.
#[derive(Debug, Clone)]
pub struct LeaderboardEntry {
    pub rank: usize,
    pub name: String,
    pub score: f64,
}

/// A leaderboard that ranks candidates.
pub struct Leaderboard {
    pub entries: Vec<LeaderboardEntry>,
}

impl Leaderboard {
    pub fn from_scorer(candidates: &[Candidate], scorer: &dyn ScoreFunction) -> Self {
        let mut scored: Vec<(String, f64)> = candidates.iter()
            .map(|c| (c.name.clone(), scorer.score(c)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let entries = scored.into_iter().enumerate().map(|(i, (name, score))| {
            LeaderboardEntry { rank: i + 1, name, score }
        }).collect();
        Self { entries }
    }

    pub fn winner(&self) -> Option<&LeaderboardEntry> {
        self.entries.first()
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(name: &str, speed: f64, accuracy: f64) -> Candidate {
        Candidate::new(name).with_score("speed", speed).with_score("accuracy", accuracy)
    }

    #[test]
    fn test_candidate_get_score() {
        let c = make_candidate("a", 0.9, 0.8);
        assert_eq!(c.get_score("speed"), Some(0.9));
        assert_eq!(c.get_score("accuracy"), Some(0.8));
        assert_eq!(c.get_score("other"), None);
    }

    #[test]
    fn test_weighted_scorer() {
        let scorer = WeightedScorer::new(vec![("speed", 0.6), ("accuracy", 0.4)]);
        let c = make_candidate("a", 1.0, 0.5);
        let score = scorer.score(&c);
        assert!((score - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_uniform() {
        let scorer = WeightedScorer::uniform(&["speed", "accuracy"]);
        let c = make_candidate("a", 1.0, 1.0);
        let score = scorer.score(&c);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_missing_criterion() {
        let scorer = WeightedScorer::new(vec![("speed", 1.0)]);
        let c = Candidate::new("x");
        assert_eq!(scorer.score(&c), 0.0);
    }

    #[test]
    fn test_pareto_dominates() {
        let p = ParetoScorer::new(vec![("speed", true), ("cost", false)]);
        let a = Candidate::new("a").with_score("speed", 10.0).with_score("cost", 5.0);
        let b = Candidate::new("b").with_score("speed", 8.0).with_score("cost", 6.0);
        assert!(p.dominates(&a, &b));
        assert!(!p.dominates(&b, &a));
    }

    #[test]
    fn test_pareto_no_dominance() {
        let p = ParetoScorer::new(vec![("speed", true), ("cost", false)]);
        let a = Candidate::new("a").with_score("speed", 10.0).with_score("cost", 10.0);
        let b = Candidate::new("b").with_score("speed", 5.0).with_score("cost", 5.0);
        assert!(!p.dominates(&a, &b)); // a has better speed but worse cost
        assert!(!p.dominates(&b, &a));
    }

    #[test]
    fn test_pareto_front() {
        let p = ParetoScorer::new(vec![("speed", true), ("cost", false)]);
        let a = Candidate::new("a").with_score("speed", 10.0).with_score("cost", 5.0);
        let b = Candidate::new("b").with_score("speed", 8.0).with_score("cost", 3.0);
        let c = Candidate::new("c").with_score("speed", 5.0).with_score("cost", 10.0);
        let front = p.pareto_front(&[a, b, c]);
        assert_eq!(front.len(), 2);
        assert!(front.contains(&0)); // a
        assert!(front.contains(&1)); // b
    }

    #[test]
    fn test_pareto_score() {
        let p = ParetoScorer::new(vec![("a", true), ("b", true)]);
        let c = Candidate::new("x").with_score("a", 3.0).with_score("b", 4.0);
        assert_eq!(p.score(&c), 7.0);
    }

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
    fn test_normalizer_constant() {
        let c1 = Candidate::new("a").with_score("x", 5.0);
        let c2 = Candidate::new("b").with_score("x", 5.0);
        let norm = ScoreNormalizer::from_candidates(&[c1.clone(), c2.clone()]);
        let n = norm.normalize(&c1);
        assert_eq!(n.get_score("x"), Some(0.0)); // constant → 0
    }

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
    fn test_scorer_name() {
        let w = WeightedScorer::new(vec![]);
        assert_eq!(w.name(), "weighted");
        let p = ParetoScorer::new(vec![]);
        assert_eq!(p.name(), "pareto");
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
    fn test_candidate_with_multiple_scores() {
        let c = Candidate::new("multi")
            .with_score("a", 1.0)
            .with_score("b", 2.0)
            .with_score("c", 3.0);
        assert_eq!(c.scores.len(), 3);
    }

    #[test]
    fn test_three_way_leaderboard() {
        let scorer = WeightedScorer::new(vec![("a", 0.5), ("b", 0.5)]);
        let c1 = Candidate::new("low").with_score("a", 0.0).with_score("b", 0.0);
        let c2 = Candidate::new("mid").with_score("a", 0.5).with_score("b", 0.5);
        let c3 = Candidate::new("high").with_score("a", 1.0).with_score("b", 1.0);
        let lb = Leaderboard::from_scorer(&[c1, c2, c3], &scorer);
        assert_eq!(lb.entries[0].name, "high");
        assert_eq!(lb.entries[2].name, "low");
    }
}
