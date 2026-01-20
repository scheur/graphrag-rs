//! Selective cache invalidation for incremental graph updates.
//!
//! This module provides intelligent cache management that invalidates
//! only the affected cache regions when graph modifications occur.

use crate::core::{DocumentId, EntityId};
use chrono::{DateTime, Utc};
use std::collections::HashSet;

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::Mutex,
};

#[cfg(feature = "incremental")]
use super::change_log::{ChangeData, ChangeRecord};
#[cfg(feature = "incremental")]
use super::types::ChangeType;



// ============================================================================
// Cache Invalidation Strategies
// ============================================================================

/// Cache invalidation strategies
#[derive(Debug, Clone)]
pub enum InvalidationStrategy {
    /// Invalidate specific cache keys
    Selective(Vec<String>),
    /// Invalidate all caches in a region
    Regional(String),
    /// Invalidate all caches
    Global,
    /// Invalidate based on entity relationships
    Relational(EntityId, u32), // entity_id, depth
}

/// Cache region affected by changes
#[derive(Debug, Clone)]
pub struct CacheRegion {
    /// Unique identifier for the cache region
    pub region_id: String,
    /// Entity IDs in this region
    pub entity_ids: HashSet<EntityId>,
    /// Relationship types in this region
    pub relationship_types: HashSet<String>,
    /// Document IDs in this region
    pub document_ids: HashSet<DocumentId>,
    /// When the region was last modified
    pub last_modified: DateTime<Utc>,
}

/// Statistics about cache invalidations
#[derive(Debug, Clone)]
pub struct InvalidationStats {
    /// Total number of invalidations performed
    pub total_invalidations: usize,
    /// Number of cache regions registered
    pub cache_regions: usize,
    /// Number of entity-to-region mappings
    pub entity_mappings: usize,
    /// Timestamp of last invalidation
    pub last_invalidation: Option<DateTime<Utc>>,
}

// ============================================================================
// Selective Invalidation Manager
// ============================================================================

/// Selective cache invalidation manager
#[cfg(feature = "incremental")]
pub struct SelectiveInvalidation {
    cache_regions: DashMap<String, CacheRegion>,
    entity_to_regions: DashMap<EntityId, HashSet<String>>,
    invalidation_log: Mutex<Vec<(DateTime<Utc>, InvalidationStrategy)>>,
}

#[cfg(feature = "incremental")]
impl Default for SelectiveInvalidation {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "incremental")]
impl SelectiveInvalidation {
    /// Creates a new selective invalidation manager
    pub fn new() -> Self {
        Self {
            cache_regions: DashMap::new(),
            entity_to_regions: DashMap::new(),
            invalidation_log: Mutex::new(Vec::new()),
        }
    }

    /// Registers a cache region for invalidation tracking
    pub fn register_cache_region(&self, region: CacheRegion) {
        let region_id = region.region_id.clone();

        // Update entity mappings
        for entity_id in &region.entity_ids {
            self.entity_to_regions
                .entry(entity_id.clone())
                .or_default()
                .insert(region_id.clone());
        }

        self.cache_regions.insert(region_id, region);
    }

    /// Determines invalidation strategies for a set of changes
    pub fn invalidate_for_changes(&self, changes: &[ChangeRecord]) -> Vec<InvalidationStrategy> {
        let mut strategies = Vec::new();
        let mut affected_regions = HashSet::new();

        for change in changes {
            match &change.change_type {
                ChangeType::EntityAdded | ChangeType::EntityUpdated | ChangeType::EntityRemoved => {
                    if let Some(entity_id) = &change.entity_id {
                        if let Some(regions) = self.entity_to_regions.get(entity_id) {
                            affected_regions.extend(regions.clone());
                        }
                        strategies.push(InvalidationStrategy::Relational(entity_id.clone(), 2));
                    }
                }
                ChangeType::RelationshipAdded
                | ChangeType::RelationshipUpdated
                | ChangeType::RelationshipRemoved => {
                    // Invalidate based on relationship endpoints
                    if let ChangeData::Relationship(rel) = &change.data {
                        strategies.push(InvalidationStrategy::Relational(rel.source.clone(), 1));
                        strategies.push(InvalidationStrategy::Relational(rel.target.clone(), 1));
                    }
                }
                _ => {
                    // For other changes, use selective invalidation
                    let cache_keys = self.generate_cache_keys_for_change(change);
                    if !cache_keys.is_empty() {
                        strategies.push(InvalidationStrategy::Selective(cache_keys));
                    }
                }
            }
        }

        // Add regional invalidation for affected regions
        for region_id in affected_regions {
            strategies.push(InvalidationStrategy::Regional(region_id));
        }

        // Log invalidation
        let mut log = self.invalidation_log.lock();
        for strategy in &strategies {
            log.push((Utc::now(), strategy.clone()));
        }

        strategies
    }

    fn generate_cache_keys_for_change(&self, change: &ChangeRecord) -> Vec<String> {
        let mut keys = Vec::new();

        // Generate cache keys based on change type and data
        match &change.change_type {
            ChangeType::EntityAdded | ChangeType::EntityUpdated => {
                if let Some(entity_id) = &change.entity_id {
                    keys.push(format!("entity:{entity_id}"));
                    keys.push(format!("entity_neighbors:{entity_id}"));
                }
            }
            ChangeType::DocumentAdded | ChangeType::DocumentUpdated => {
                if let Some(doc_id) = &change.document_id {
                    keys.push(format!("document:{doc_id}"));
                    keys.push(format!("document_chunks:{doc_id}"));
                }
            }
            ChangeType::EmbeddingAdded | ChangeType::EmbeddingUpdated => {
                if let Some(entity_id) = &change.entity_id {
                    keys.push(format!("embedding:{entity_id}"));
                    keys.push(format!("similarity:{entity_id}"));
                }
            }
            _ => {}
        }

        keys
    }

    /// Gets statistics about cache invalidations
    pub fn get_invalidation_stats(&self) -> InvalidationStats {
        let log = self.invalidation_log.lock();

        InvalidationStats {
            total_invalidations: log.len(),
            cache_regions: self.cache_regions.len(),
            entity_mappings: self.entity_to_regions.len(),
            last_invalidation: log.last().map(|(time, _)| *time),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "incremental")]
    #[test]
    fn test_selective_invalidation_creation() {
        let invalidation = SelectiveInvalidation::new();
        let stats = invalidation.get_invalidation_stats();
        assert_eq!(stats.total_invalidations, 0);
        assert_eq!(stats.cache_regions, 0);
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_register_cache_region() {
        let invalidation = SelectiveInvalidation::new();
        let mut entity_ids = HashSet::new();
        entity_ids.insert(EntityId::new("test".to_string()));

        let region = CacheRegion {
            region_id: "test_region".to_string(),
            entity_ids,
            relationship_types: HashSet::new(),
            document_ids: HashSet::new(),
            last_modified: Utc::now(),
        };

        invalidation.register_cache_region(region);
        let stats = invalidation.get_invalidation_stats();
        assert_eq!(stats.cache_regions, 1);
        assert_eq!(stats.entity_mappings, 1);
    }
}
