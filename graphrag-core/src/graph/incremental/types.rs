//! Core types and identifiers for incremental graph operations.
//!
//! This module contains fundamental types used throughout the incremental
//! update system: unique identifiers, operation enums, and status types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
#[cfg(not(feature = "incremental"))]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "incremental")]
use uuid::Uuid;

// ============================================================================
// Unique Identifiers
// ============================================================================

/// Unique identifier for update operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UpdateId(String);

#[cfg(not(feature = "incremental"))]
static UPDATE_COUNTER: AtomicU64 = AtomicU64::new(0);

impl UpdateId {
    /// Creates a new unique update identifier
    pub fn new() -> Self {
        #[cfg(feature = "incremental")]
        {
            Self(Uuid::new_v4().to_string())
        }
        #[cfg(not(feature = "incremental"))]
        {
            let nanos = Utc::now().timestamp_nanos_opt().unwrap_or(0);
            let counter = UPDATE_COUNTER.fetch_add(1, Ordering::SeqCst);
            Self(format!("update_{}_{}", nanos, counter))
        }
    }

    /// Creates an update identifier from an existing string
    pub fn from_string(id: String) -> Self {
        Self(id)
    }

    /// Returns the update ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for UpdateId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for UpdateId {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction identifier for atomic operations
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransactionId(String);

impl TransactionId {
    /// Creates a new unique transaction identifier
    pub fn new() -> Self {
        #[cfg(feature = "incremental")]
        {
            Self(Uuid::new_v4().to_string())
        }
        #[cfg(not(feature = "incremental"))]
        {
            Self(format!(
                "tx_{}",
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ))
        }
    }

    /// Returns the transaction ID as a string slice
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TransactionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for TransactionId {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Operation and Change Enums
// ============================================================================

/// Types of changes that can occur in the graph
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ChangeType {
    /// An entity was added to the graph
    EntityAdded,
    /// An existing entity was updated
    EntityUpdated,
    /// An entity was removed from the graph
    EntityRemoved,
    /// A relationship was added to the graph
    RelationshipAdded,
    /// An existing relationship was updated
    RelationshipUpdated,
    /// A relationship was removed from the graph
    RelationshipRemoved,
    /// A document was added
    DocumentAdded,
    /// An existing document was updated
    DocumentUpdated,
    /// A document was removed
    DocumentRemoved,
    /// A text chunk was added
    ChunkAdded,
    /// An existing text chunk was updated
    ChunkUpdated,
    /// A text chunk was removed
    ChunkRemoved,
    /// An embedding was added
    EmbeddingAdded,
    /// An existing embedding was updated
    EmbeddingUpdated,
    /// An embedding was removed
    EmbeddingRemoved,
}

/// Operations that can be performed on graph elements
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Operation {
    /// Insert a new item
    Insert,
    /// Update an existing item
    Update,
    /// Delete an item
    Delete,
    /// Insert or update (upsert) an item
    Upsert,
}

/// Status of a delta operation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DeltaStatus {
    /// Delta is pending application
    Pending,
    /// Delta has been applied but not committed
    Applied,
    /// Delta has been committed
    Committed,
    /// Delta has been rolled back
    RolledBack,
    /// Delta failed with error message
    Failed {
        /// Error message describing the failure
        error: String,
    },
}

// ============================================================================
// Statistics and Reports
// ============================================================================

/// Graph statistics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStatistics {
    /// Total number of nodes (entities)
    pub node_count: usize,
    /// Total number of edges (relationships)
    pub edge_count: usize,
    /// Average degree of nodes
    pub average_degree: f64,
    /// Maximum degree of any node
    pub max_degree: usize,
    /// Number of connected components
    pub connected_components: usize,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// When statistics were last updated
    pub last_updated: DateTime<Utc>,
}

/// Consistency validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyReport {
    /// Whether the graph is consistent
    pub is_consistent: bool,
    /// Entities with no relationships
    pub orphaned_entities: Vec<crate::core::EntityId>,
    /// Relationships referencing non-existent entities
    pub broken_relationships: Vec<(crate::core::EntityId, crate::core::EntityId, String)>,
    /// Entities missing embeddings
    pub missing_embeddings: Vec<crate::core::EntityId>,
    /// When validation was performed
    pub validation_time: DateTime<Utc>,
    /// Total number of issues found
    pub issues_found: usize,
}

/// Comprehensive statistics for incremental operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncrementalStatistics {
    /// Total number of update operations
    pub total_updates: usize,
    /// Number of successful updates
    pub successful_updates: usize,
    /// Number of failed updates
    pub failed_updates: usize,
    /// Number of entities added
    pub entities_added: usize,
    /// Number of entities updated
    pub entities_updated: usize,
    /// Number of entities removed
    pub entities_removed: usize,
    /// Number of relationships added
    pub relationships_added: usize,
    /// Number of relationships updated
    pub relationships_updated: usize,
    /// Number of relationships removed
    pub relationships_removed: usize,
    /// Number of conflicts resolved
    pub conflicts_resolved: usize,
    /// Number of cache invalidations performed
    pub cache_invalidations: usize,
    /// Average update time in milliseconds
    pub average_update_time_ms: f64,
    /// Peak updates per second achieved
    pub peak_updates_per_second: f64,
    /// Current size of the change log
    pub current_change_log_size: usize,
    /// Current number of active deltas
    pub current_delta_count: usize,
}

impl IncrementalStatistics {
    /// Creates an empty statistics instance
    pub fn empty() -> Self {
        Self {
            total_updates: 0,
            successful_updates: 0,
            failed_updates: 0,
            entities_added: 0,
            entities_updated: 0,
            entities_removed: 0,
            relationships_added: 0,
            relationships_updated: 0,
            relationships_removed: 0,
            conflicts_resolved: 0,
            cache_invalidations: 0,
            average_update_time_ms: 0.0,
            peak_updates_per_second: 0.0,
            current_change_log_size: 0,
            current_delta_count: 0,
        }
    }

    /// Prints statistics to stdout in a formatted way
    pub fn print(&self) {
        println!("🔄 Incremental Updates Statistics");
        println!("  Total updates: {}", self.total_updates);
        println!(
            "  Successful: {} ({:.1}%)",
            self.successful_updates,
            if self.total_updates > 0 {
                (self.successful_updates as f64 / self.total_updates as f64) * 100.0
            } else {
                0.0
            }
        );
        println!("  Failed: {}", self.failed_updates);
        println!(
            "  Entities: +{} ~{} -{}",
            self.entities_added, self.entities_updated, self.entities_removed
        );
        println!(
            "  Relationships: +{} ~{} -{}",
            self.relationships_added, self.relationships_updated, self.relationships_removed
        );
        println!("  Conflicts resolved: {}", self.conflicts_resolved);
        println!("  Cache invalidations: {}", self.cache_invalidations);
        println!("  Avg update time: {:.2}ms", self.average_update_time_ms);
        println!("  Peak updates/sec: {:.1}", self.peak_updates_per_second);
        println!("  Change log size: {}", self.current_change_log_size);
        println!("  Active deltas: {}", self.current_delta_count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_id_generation() {
        let id1 = UpdateId::new();
        let id2 = UpdateId::new();
        assert_ne!(id1.as_str(), id2.as_str());
    }

    #[test]
    fn test_transaction_id_generation() {
        let tx1 = TransactionId::new();
        let tx2 = TransactionId::new();
        assert_ne!(tx1.as_str(), tx2.as_str());
    }

    #[test]
    fn test_statistics_creation() {
        let stats = IncrementalStatistics::empty();
        assert_eq!(stats.total_updates, 0);
        assert_eq!(stats.successful_updates, 0);
    }
}
