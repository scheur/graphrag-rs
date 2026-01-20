//! Incremental PageRank calculation for graph updates.
//!
//! This module provides efficient incremental PageRank computation
//! that avoids full graph recomputation when possible.

#[cfg(feature = "incremental")]
use crate::core::{EntityId, KnowledgeGraph, Result};
#[cfg(feature = "incremental")]
use chrono::{DateTime, Utc};
#[cfg(feature = "incremental")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "incremental")]
use std::time::Instant;

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::RwLock,
};

// ============================================================================
// Incremental PageRank
// ============================================================================

/// Incremental PageRank calculator for efficient updates
#[cfg(feature = "incremental")]
#[allow(dead_code)]
pub struct IncrementalPageRank {
    scores: DashMap<EntityId, f64>,
    adjacency_changes: DashMap<EntityId, Vec<(EntityId, f64)>>, // Node -> [(neighbor, weight)]
    damping_factor: f64,
    tolerance: f64,
    max_iterations: usize,
    last_full_computation: DateTime<Utc>,
    incremental_threshold: usize, // Number of changes before full recomputation
    pending_changes: RwLock<usize>,
}

#[cfg(feature = "incremental")]
impl IncrementalPageRank {
    /// Creates a new incremental PageRank calculator
    pub fn new(damping_factor: f64, tolerance: f64, max_iterations: usize) -> Self {
        Self {
            scores: DashMap::new(),
            adjacency_changes: DashMap::new(),
            damping_factor,
            tolerance,
            max_iterations,
            last_full_computation: Utc::now(),
            incremental_threshold: 1000,
            pending_changes: RwLock::new(0),
        }
    }

    /// Update PageRank incrementally for a specific subgraph
    pub async fn update_incremental(
        &self,
        changed_entities: &[EntityId],
        graph: &KnowledgeGraph,
    ) -> Result<()> {
        let start = Instant::now();

        // If too many changes accumulated, do full recomputation
        {
            let pending = *self.pending_changes.read();
            if pending > self.incremental_threshold {
                return self.full_recomputation(graph).await;
            }
        }

        // Incremental update for changed entities and their neighborhoods
        let mut affected_entities = HashSet::new();

        // Add changed entities and their neighbors (2-hop neighborhood)
        for entity_id in changed_entities {
            affected_entities.insert(entity_id.clone());

            // Add direct neighbors
            for (neighbor, _) in graph.get_neighbors(entity_id) {
                affected_entities.insert(neighbor.id.clone());

                // Add second-hop neighbors
                for (second_hop, _) in graph.get_neighbors(&neighbor.id) {
                    affected_entities.insert(second_hop.id.clone());
                }
            }
        }

        // Perform localized PageRank computation
        self.localized_pagerank(&affected_entities, graph).await?;

        // Reset pending changes counter
        *self.pending_changes.write() = 0;

        let duration = start.elapsed();
        println!(
            "🔄 Incremental PageRank update completed in {:?} for {} entities",
            duration,
            affected_entities.len()
        );

        Ok(())
    }

    /// Perform full PageRank recomputation
    async fn full_recomputation(&self, graph: &KnowledgeGraph) -> Result<()> {
        let start = Instant::now();

        // Build adjacency matrix
        let entities: Vec<EntityId> = graph.entities().map(|e| e.id.clone()).collect();
        let n = entities.len();

        if n == 0 {
            return Ok(());
        }

        // Initialize scores
        let initial_score = 1.0 / n as f64;
        for entity_id in &entities {
            self.scores.insert(entity_id.clone(), initial_score);
        }

        // Power iteration
        for iteration in 0..self.max_iterations {
            let mut new_scores = HashMap::new();
            let mut max_diff: f64 = 0.0;

            for entity_id in &entities {
                let mut score = (1.0 - self.damping_factor) / n as f64;

                // Sum contributions from incoming links
                for other_entity in &entities {
                    if let Some(weight) = self.get_edge_weight(other_entity, entity_id, graph) {
                        let other_score = self
                            .scores
                            .get(other_entity)
                            .map(|s| *s.value())
                            .unwrap_or(initial_score);
                        let out_degree = self.get_out_degree(other_entity, graph);

                        if out_degree > 0.0 {
                            score += self.damping_factor * other_score * weight / out_degree;
                        }
                    }
                }

                let old_score = self
                    .scores
                    .get(entity_id)
                    .map(|s| *s.value())
                    .unwrap_or(initial_score);
                let diff = (score - old_score).abs();
                max_diff = max_diff.max(diff);

                new_scores.insert(entity_id.clone(), score);
            }

            // Update scores
            for (entity_id, score) in new_scores {
                self.scores.insert(entity_id, score);
            }

            // Check convergence
            if max_diff < self.tolerance {
                println!(
                    "🎯 PageRank converged after {} iterations (diff: {:.6})",
                    iteration + 1,
                    max_diff
                );
                break;
            }
        }

        let duration = start.elapsed();
        println!(
            "🔄 Full PageRank recomputation completed in {duration:?} for {n} entities"
        );

        Ok(())
    }

    /// Perform localized PageRank computation for a subset of entities
    async fn localized_pagerank(
        &self,
        entities: &HashSet<EntityId>,
        graph: &KnowledgeGraph,
    ) -> Result<()> {
        let entity_vec: Vec<EntityId> = entities.iter().cloned().collect();
        let n = entity_vec.len();

        if n == 0 {
            return Ok(());
        }

        // Localized power iteration
        for _iteration in 0..self.max_iterations {
            let mut max_diff: f64 = 0.0;

            for entity_id in &entity_vec {
                let mut score = (1.0 - self.damping_factor) / n as f64;

                // Only consider links within the subset for localized computation
                for other_entity in &entity_vec {
                    if let Some(weight) = self.get_edge_weight(other_entity, entity_id, graph) {
                        let other_score = self
                            .scores
                            .get(other_entity)
                            .map(|s| *s.value())
                            .unwrap_or(1.0 / n as f64);
                        let out_degree =
                            self.get_localized_out_degree(other_entity, entities, graph);

                        if out_degree > 0.0 {
                            score += self.damping_factor * other_score * weight / out_degree;
                        }
                    }
                }

                let old_score = self
                    .scores
                    .get(entity_id)
                    .map(|s| *s.value())
                    .unwrap_or(1.0 / n as f64);
                let diff = (score - old_score).abs();
                max_diff = max_diff.max(diff);

                self.scores.insert(entity_id.clone(), score);
            }

            // Check convergence
            if max_diff < self.tolerance {
                break;
            }
        }

        Ok(())
    }

    fn get_edge_weight(
        &self,
        from: &EntityId,
        to: &EntityId,
        graph: &KnowledgeGraph,
    ) -> Option<f64> {
        // Check if there's a relationship between entities
        for (neighbor, relationship) in graph.get_neighbors(from) {
            if neighbor.id == *to {
                return Some(relationship.confidence as f64);
            }
        }
        None
    }

    fn get_out_degree(&self, entity_id: &EntityId, graph: &KnowledgeGraph) -> f64 {
        graph
            .get_neighbors(entity_id)
            .iter()
            .map(|(_, rel)| rel.confidence as f64)
            .sum()
    }

    fn get_localized_out_degree(
        &self,
        entity_id: &EntityId,
        subset: &HashSet<EntityId>,
        graph: &KnowledgeGraph,
    ) -> f64 {
        graph
            .get_neighbors(entity_id)
            .iter()
            .filter(|(neighbor, _)| subset.contains(&neighbor.id))
            .map(|(_, rel)| rel.confidence as f64)
            .sum()
    }

    /// Get PageRank score for an entity
    pub fn get_score(&self, entity_id: &EntityId) -> Option<f64> {
        self.scores.get(entity_id).map(|s| *s.value())
    }

    /// Get top-k entities by PageRank score
    pub fn get_top_entities(&self, k: usize) -> Vec<(EntityId, f64)> {
        let mut entities: Vec<(EntityId, f64)> = self
            .scores
            .iter()
            .map(|entry| (entry.key().clone(), *entry.value()))
            .collect();

        entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entities.truncate(k);
        entities
    }

    /// Record a graph change for incremental updates
    pub fn record_change(&self, _entity_id: EntityId) {
        *self.pending_changes.write() += 1;
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "incremental")]
    use super::*;

    #[cfg(feature = "incremental")]
    #[test]
    fn test_incremental_pagerank_creation() {
        let pr = IncrementalPageRank::new(0.85, 1e-6, 100);
        assert_eq!(pr.damping_factor, 0.85);
        assert_eq!(pr.tolerance, 1e-6);
        assert_eq!(pr.max_iterations, 100);
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_get_score_empty() {
        let pr = IncrementalPageRank::new(0.85, 1e-6, 100);
        let entity_id = EntityId::new("test".to_string());
        assert!(pr.get_score(&entity_id).is_none());
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_get_top_entities_empty() {
        let pr = IncrementalPageRank::new(0.85, 1e-6, 100);
        let top = pr.get_top_entities(10);
        assert!(top.is_empty());
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_record_change() {
        let pr = IncrementalPageRank::new(0.85, 1e-6, 100);
        pr.record_change(EntityId::new("test".to_string()));
        assert_eq!(*pr.pending_changes.read(), 1);
    }
}
