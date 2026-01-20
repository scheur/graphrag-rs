//! Monitoring and metrics for incremental graph operations.
//!
//! This module provides performance tracking, operation logging,
//! and metrics collection for monitoring incremental updates.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::{Mutex, RwLock},
};

use super::types::UpdateId;

// ============================================================================
// Metrics and Logging Types
// ============================================================================

/// Metric for tracking update operations
#[derive(Debug, Clone)]
pub struct UpdateMetric {
    /// Name of the metric
    pub name: String,
    /// Metric value
    pub value: f64,
    /// When the metric was recorded
    pub timestamp: DateTime<Utc>,
    /// Tags for categorizing the metric
    pub tags: HashMap<String, String>,
}

/// Log entry for an operation
#[derive(Debug, Clone)]
pub struct OperationLog {
    /// Unique operation identifier
    pub operation_id: UpdateId,
    /// Type of operation performed
    pub operation_type: String,
    /// When the operation started
    pub start_time: Instant,
    /// When the operation ended
    pub end_time: Option<Instant>,
    /// Whether the operation succeeded
    pub success: Option<bool>,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Number of entities affected
    pub affected_entities: usize,
    /// Number of relationships affected
    pub affected_relationships: usize,
}

/// Performance statistics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total number of operations performed
    pub total_operations: u64,
    /// Number of successful operations
    pub successful_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Average time per operation
    pub average_operation_time: Duration,
    /// Peak throughput in operations per second
    pub peak_operations_per_second: f64,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f64,
    /// Conflict resolution rate (0.0 to 1.0)
    pub conflict_resolution_rate: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            average_operation_time: Duration::from_millis(0),
            peak_operations_per_second: 0.0,
            cache_hit_rate: 0.0,
            conflict_resolution_rate: 0.0,
        }
    }
}

// ============================================================================
// Update Monitor
// ============================================================================

/// Monitor for tracking update operations and performance
#[cfg(feature = "incremental")]
pub struct UpdateMonitor {
    metrics: DashMap<String, UpdateMetric>,
    operations_log: Mutex<Vec<OperationLog>>,
    performance_stats: RwLock<PerformanceStats>,
}

#[cfg(feature = "incremental")]
impl Default for UpdateMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "incremental")]
impl UpdateMonitor {
    /// Creates a new update monitor
    pub fn new() -> Self {
        Self {
            metrics: DashMap::new(),
            operations_log: Mutex::new(Vec::new()),
            performance_stats: RwLock::new(PerformanceStats::default()),
        }
    }

    /// Starts tracking a new operation and returns its ID
    pub fn start_operation(&self, operation_type: &str) -> UpdateId {
        let operation_id = UpdateId::new();
        let log_entry = OperationLog {
            operation_id: operation_id.clone(),
            operation_type: operation_type.to_string(),
            start_time: Instant::now(),
            end_time: None,
            success: None,
            error_message: None,
            affected_entities: 0,
            affected_relationships: 0,
        };

        self.operations_log.lock().push(log_entry);
        operation_id
    }

    /// Marks an operation as complete with results
    pub fn complete_operation(
        &self,
        operation_id: &UpdateId,
        success: bool,
        error: Option<String>,
        affected_entities: usize,
        affected_relationships: usize,
    ) {
        let mut log = self.operations_log.lock();
        if let Some(entry) = log.iter_mut().find(|e| &e.operation_id == operation_id) {
            entry.end_time = Some(Instant::now());
            entry.success = Some(success);
            entry.error_message = error;
            entry.affected_entities = affected_entities;
            entry.affected_relationships = affected_relationships;
        }
        drop(log);

        // Update performance stats
        self.update_performance_stats();
    }

    fn update_performance_stats(&self) {
        let log = self.operations_log.lock();
        let completed_ops: Vec<_> = log
            .iter()
            .filter(|op| op.end_time.is_some() && op.success.is_some())
            .collect();

        if completed_ops.is_empty() {
            return;
        }

        let mut stats = self.performance_stats.write();
        stats.total_operations = completed_ops.len() as u64;
        stats.successful_operations = completed_ops
            .iter()
            .filter(|op| op.success == Some(true))
            .count() as u64;
        stats.failed_operations = stats.total_operations - stats.successful_operations;

        // Calculate average operation time
        let total_time: Duration = completed_ops
            .iter()
            .filter_map(|op| op.end_time.map(|end| end.duration_since(op.start_time)))
            .sum();

        if !completed_ops.is_empty() {
            stats.average_operation_time = total_time / completed_ops.len() as u32;
        }
    }

    /// Records a metric with tags
    pub fn record_metric(&self, name: &str, value: f64, tags: HashMap<String, String>) {
        let metric = UpdateMetric {
            name: name.to_string(),
            value,
            timestamp: Utc::now(),
            tags,
        };
        self.metrics.insert(name.to_string(), metric);
    }

    /// Gets the current performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_stats.read().clone()
    }

    /// Gets the most recent operations up to the specified limit
    pub fn get_recent_operations(&self, limit: usize) -> Vec<OperationLog> {
        let log = self.operations_log.lock();
        log.iter().rev().take(limit).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_stats_default() {
        let stats = PerformanceStats::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.successful_operations, 0);
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_update_monitor_creation() {
        let monitor = UpdateMonitor::new();
        let stats = monitor.get_performance_stats();
        assert_eq!(stats.total_operations, 0);
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_start_and_complete_operation() {
        let monitor = UpdateMonitor::new();
        let op_id = monitor.start_operation("test_op");
        monitor.complete_operation(&op_id, true, None, 5, 3);

        let stats = monitor.get_performance_stats();
        assert_eq!(stats.total_operations, 1);
        assert_eq!(stats.successful_operations, 1);

        let recent = monitor.get_recent_operations(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].affected_entities, 5);
        assert_eq!(recent[0].affected_relationships, 3);
    }
}
