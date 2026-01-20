//! Incremental graph update system with ACID-like guarantees.
//!
//! This module provides incremental update capabilities for knowledge graphs,
//! including conflict resolution, batch processing, cache invalidation, and
//! monitoring.
//!
//! # Architecture
//!
//! The incremental system is organized into focused submodules:
//!
//! - [`types`] - Core types, enums, and identifiers
//! - [`change_log`] - Change tracking and rollback support
//! - [`conflict`] - Conflict detection and resolution strategies
//! - [`cache`] - Selective cache invalidation
//! - [`monitor`] - Performance monitoring and operation logging
//! - [`manager`] - Main incremental graph manager
//! - [`pagerank`] - Incremental PageRank computation
//! - [`batch`] - Batch processing for high throughput
//! - [`store`] - Storage traits and implementations
//!
//! # Feature Gates
//!
//! Most functionality requires the `incremental` feature. Basic types are
//! available without the feature flag.

// Submodules
pub mod batch;
pub mod cache;
pub mod change_log;
pub mod conflict;
pub mod manager;
pub mod monitor;
pub mod pagerank;
pub mod store;
pub mod types;

// Re-export core types (always available)
pub use types::{
    ChangeType, ConsistencyReport, DeltaStatus, GraphStatistics, IncrementalStatistics, Operation,
    TransactionId, UpdateId,
};

// Re-export change log types (always available)
pub use change_log::{ChangeData, ChangeRecord, Document, GraphDelta, RollbackData};

// Re-export conflict resolution (always available)
pub use conflict::{Conflict, ConflictResolution, ConflictResolver, ConflictStrategy, ConflictType};

// Re-export cache invalidation types (always available)
pub use cache::{CacheRegion, InvalidationStats, InvalidationStrategy};

// Re-export monitoring types (always available)
pub use monitor::{OperationLog, PerformanceStats, UpdateMetric};

// Re-export manager config (always available)
pub use manager::IncrementalConfig;

// Re-export store types (always available)
pub use store::{ChangeEvent, ChangeEventType, IncrementalGraphStore};

// Feature-gated exports (require "incremental" feature)
#[cfg(feature = "incremental")]
pub use cache::SelectiveInvalidation;

#[cfg(feature = "incremental")]
pub use monitor::UpdateMonitor;

#[cfg(feature = "incremental")]
pub use manager::IncrementalGraphManager;

#[cfg(feature = "incremental")]
pub use pagerank::IncrementalPageRank;

// Re-export batch metrics (always available)
pub use batch::BatchMetrics;

#[cfg(feature = "incremental")]
pub use batch::BatchProcessor;

#[cfg(feature = "incremental")]
pub use store::ProductionGraphStore;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify core types are accessible
        let _update_id = UpdateId::new();
        let _tx_id = TransactionId::new();

        // Verify enums are accessible
        let _change = ChangeType::EntityAdded;
        let _op = Operation::Add;
        let _status = DeltaStatus::Pending;

        // Verify conflict types
        let _strategy = ConflictStrategy::KeepExisting;
    }

    #[test]
    fn test_statistics_creation() {
        let stats = GraphStatistics {
            entity_count: 100,
            relationship_count: 50,
            document_count: 10,
            chunk_count: 30,
        };

        assert_eq!(stats.entity_count, 100);
        assert_eq!(stats.relationship_count, 50);
    }
}
