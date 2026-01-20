//! Incremental graph manager for coordinating updates.
//!
//! This module provides the main entry point for incremental graph
//! operations, managing change logs, caches, and conflict resolution.

use crate::core::{DocumentId, Entity, EntityId, KnowledgeGraph, Result};
use chrono::Utc;
use std::collections::HashMap;

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::RwLock,
    std::sync::Arc,
};

use super::change_log::{ChangeData, ChangeRecord};
use super::conflict::ConflictStrategy;
use super::types::{ChangeType, IncrementalStatistics, Operation, UpdateId};

#[cfg(feature = "incremental")]
use super::cache::SelectiveInvalidation;
#[cfg(feature = "incremental")]
use super::conflict::ConflictResolver;
#[cfg(feature = "incremental")]
use super::monitor::UpdateMonitor;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for incremental operations
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Maximum number of changes to keep in the log
    pub max_change_log_size: usize,
    /// Maximum number of changes in a single delta
    pub max_delta_size: usize,
    /// Default conflict resolution strategy
    pub conflict_strategy: ConflictStrategy,
    /// Whether to enable performance monitoring
    pub enable_monitoring: bool,
    /// Cache invalidation strategy name
    pub cache_invalidation_strategy: String,
    /// Default batch size for batch operations
    pub batch_size: usize,
    /// Maximum number of concurrent operations
    pub max_concurrent_operations: usize,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        Self {
            max_change_log_size: 10000,
            max_delta_size: 1000,
            conflict_strategy: ConflictStrategy::Merge,
            enable_monitoring: true,
            cache_invalidation_strategy: "selective".to_string(),
            batch_size: 100,
            max_concurrent_operations: 10,
        }
    }
}

// ============================================================================
// Incremental Graph Manager (Feature-gated version)
// ============================================================================

/// Comprehensive incremental graph manager with production features
#[cfg(feature = "incremental")]
pub struct IncrementalGraphManager {
    graph: Arc<RwLock<KnowledgeGraph>>,
    change_log: DashMap<UpdateId, ChangeRecord>,
    deltas: DashMap<UpdateId, super::change_log::GraphDelta>,
    cache_invalidation: Arc<SelectiveInvalidation>,
    conflict_resolver: Arc<ConflictResolver>,
    monitor: Arc<UpdateMonitor>,
    config: IncrementalConfig,
}

#[cfg(feature = "incremental")]
impl IncrementalGraphManager {
    /// Creates a new incremental graph manager with feature-gated capabilities
    pub fn new(graph: KnowledgeGraph, config: IncrementalConfig) -> Self {
        Self {
            graph: Arc::new(RwLock::new(graph)),
            change_log: DashMap::new(),
            deltas: DashMap::new(),
            cache_invalidation: Arc::new(SelectiveInvalidation::new()),
            conflict_resolver: Arc::new(ConflictResolver::new(config.conflict_strategy.clone())),
            monitor: Arc::new(UpdateMonitor::new()),
            config,
        }
    }

    /// Sets a custom conflict resolver for the manager
    pub fn with_conflict_resolver(mut self, resolver: ConflictResolver) -> Self {
        self.conflict_resolver = Arc::new(resolver);
        self
    }

    /// Get a read-only reference to the knowledge graph
    pub fn graph(&self) -> Arc<RwLock<KnowledgeGraph>> {
        Arc::clone(&self.graph)
    }

    /// Get the conflict resolver
    pub fn conflict_resolver(&self) -> Arc<ConflictResolver> {
        Arc::clone(&self.conflict_resolver)
    }

    /// Get the update monitor
    pub fn monitor(&self) -> Arc<UpdateMonitor> {
        Arc::clone(&self.monitor)
    }

    /// Gets comprehensive statistics about incremental operations
    pub fn get_statistics(&self) -> IncrementalStatistics {
        let perf_stats = self.monitor.get_performance_stats();
        let invalidation_stats = self.cache_invalidation.get_invalidation_stats();

        // Calculate entity/relationship statistics from change log
        let mut entity_stats = (0, 0, 0); // added, updated, removed
        let mut relationship_stats = (0, 0, 0);
        let conflicts_resolved = 0;

        for change in self.change_log.iter() {
            match change.value().change_type {
                ChangeType::EntityAdded => entity_stats.0 += 1,
                ChangeType::EntityUpdated => entity_stats.1 += 1,
                ChangeType::EntityRemoved => entity_stats.2 += 1,
                ChangeType::RelationshipAdded => relationship_stats.0 += 1,
                ChangeType::RelationshipUpdated => relationship_stats.1 += 1,
                ChangeType::RelationshipRemoved => relationship_stats.2 += 1,
                _ => {}
            }
        }

        IncrementalStatistics {
            total_updates: perf_stats.total_operations as usize,
            successful_updates: perf_stats.successful_operations as usize,
            failed_updates: perf_stats.failed_operations as usize,
            entities_added: entity_stats.0,
            entities_updated: entity_stats.1,
            entities_removed: entity_stats.2,
            relationships_added: relationship_stats.0,
            relationships_updated: relationship_stats.1,
            relationships_removed: relationship_stats.2,
            conflicts_resolved,
            cache_invalidations: invalidation_stats.total_invalidations,
            average_update_time_ms: perf_stats.average_operation_time.as_millis() as f64,
            peak_updates_per_second: perf_stats.peak_operations_per_second,
            current_change_log_size: self.change_log.len(),
            current_delta_count: self.deltas.len(),
        }
    }

    /// Basic entity upsert (feature-gated version)
    pub fn basic_upsert_entity(&mut self, entity: Entity) -> Result<UpdateId> {
        let update_id = UpdateId::new();
        let operation_id = self.monitor.start_operation("upsert_entity");
        let mut graph = self.graph.write();

        match graph.add_entity(entity.clone()) {
            Ok(_) => {
                let ent_id = entity.id.clone();
                let change = self.create_change_record(
                    ChangeType::EntityAdded,
                    Operation::Upsert,
                    ChangeData::Entity(entity),
                    Some(ent_id),
                    None,
                );
                self.change_log.insert(change.change_id.clone(), change);
                self.monitor
                    .complete_operation(&operation_id, true, None, 1, 0);
                Ok(update_id)
            }
            Err(e) => {
                self.monitor.complete_operation(
                    &operation_id,
                    false,
                    Some(e.to_string()),
                    0,
                    0,
                );
                Err(e)
            }
        }
    }
}

// ============================================================================
// Incremental Graph Manager (Non-feature version)
// ============================================================================

#[cfg(not(feature = "incremental"))]
/// Incremental graph manager (simplified version without incremental feature)
pub struct IncrementalGraphManager {
    graph: KnowledgeGraph,
    change_log: Vec<ChangeRecord>,
    config: IncrementalConfig,
}

#[cfg(not(feature = "incremental"))]
impl IncrementalGraphManager {
    /// Creates a new incremental graph manager without advanced features
    pub fn new(graph: KnowledgeGraph, config: IncrementalConfig) -> Self {
        Self {
            graph,
            change_log: Vec::new(),
            config,
        }
    }

    /// Gets a reference to the knowledge graph
    pub fn graph(&self) -> &KnowledgeGraph {
        &self.graph
    }

    /// Gets a mutable reference to the knowledge graph
    pub fn graph_mut(&mut self) -> &mut KnowledgeGraph {
        &mut self.graph
    }

    /// Gets basic statistics about incremental operations (non-feature version)
    pub fn get_statistics(&self) -> IncrementalStatistics {
        let mut stats = IncrementalStatistics::empty();
        stats.current_change_log_size = self.change_log.len();

        for change in &self.change_log {
            match change.change_type {
                ChangeType::EntityAdded => stats.entities_added += 1,
                ChangeType::EntityUpdated => stats.entities_updated += 1,
                ChangeType::EntityRemoved => stats.entities_removed += 1,
                ChangeType::RelationshipAdded => stats.relationships_added += 1,
                ChangeType::RelationshipUpdated => stats.relationships_updated += 1,
                ChangeType::RelationshipRemoved => stats.relationships_removed += 1,
                _ => {}
            }
        }

        stats.total_updates = self.change_log.len();
        stats.successful_updates = self.change_log.len(); // Assume all succeeded in basic mode
        stats
    }

    /// Basic entity upsert (non-feature version)
    pub fn basic_upsert_entity(&mut self, entity: Entity) -> Result<UpdateId> {
        let update_id = UpdateId::new();
        self.graph.add_entity(entity.clone())?;
        
        // Capture ID before moving `entity` into ChangeData
        let ent_id = entity.id.clone();
        let change = self.create_change_record(
            ChangeType::EntityAdded,
            Operation::Upsert,
            ChangeData::Entity(entity),
            Some(ent_id),
            None,
        );
        self.change_log.push(change);
        Ok(update_id)
    }
}

// ============================================================================
// Common Implementation
// ============================================================================

impl IncrementalGraphManager {
    /// Create a new change record
    pub fn create_change_record(
        &self,
        change_type: ChangeType,
        operation: Operation,
        change_data: ChangeData,
        entity_id: Option<EntityId>,
        document_id: Option<DocumentId>,
    ) -> ChangeRecord {
        ChangeRecord {
            change_id: UpdateId::new(),
            timestamp: Utc::now(),
            change_type,
            entity_id,
            document_id,
            operation,
            data: change_data,
            metadata: HashMap::new(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &IncrementalConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incremental_config_default() {
        let config = IncrementalConfig::default();
        assert_eq!(config.max_change_log_size, 10000);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_basic_entity_upsert() {
        let graph = KnowledgeGraph::new();
        let config = IncrementalConfig::default();
        let mut manager = IncrementalGraphManager::new(graph, config);

        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        let result = manager.basic_upsert_entity(entity);
        assert!(result.is_ok());

        let stats = manager.get_statistics();
        assert_eq!(stats.entities_added, 1);
    }
}
