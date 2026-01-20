//! Batch processing for high-throughput incremental updates.
//!
//! This module provides efficient batch processing of graph modifications
//! to maximize throughput and minimize overhead.

#[cfg(feature = "incremental")]
use crate::core::{GraphRAGError, Result};
#[cfg(feature = "incremental")]
use super::types::ChangeType;
use chrono::{DateTime, Utc};
use std::time::{Duration, Instant};

#[cfg(feature = "incremental")]
use {
    dashmap::DashMap,
    parking_lot::RwLock,
    std::sync::Arc,
    tokio::sync::Semaphore,
    uuid::Uuid,
};

use super::change_log::ChangeRecord;


// ============================================================================
// Batch Metrics
// ============================================================================

/// Batch metrics for monitoring
#[derive(Debug, Clone)]
pub struct BatchMetrics {
    /// Total number of batches processed
    pub total_batches_processed: u64,
    /// Total number of changes processed across all batches
    pub total_changes_processed: u64,
    /// Average size of batches
    pub average_batch_size: f64,
    /// Average time to process a batch
    pub average_processing_time: Duration,
    /// Throughput in changes per second
    pub throughput_per_second: f64,
    /// Timestamp of last batch processed
    pub last_batch_processed: Option<DateTime<Utc>>,
}

impl Default for BatchMetrics {
    fn default() -> Self {
        Self {
            total_batches_processed: 0,
            total_changes_processed: 0,
            average_batch_size: 0.0,
            average_processing_time: Duration::from_millis(0),
            throughput_per_second: 0.0,
            last_batch_processed: None,
        }
    }
}

// ============================================================================
// Pending Batch
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PendingBatch {
    changes: Vec<ChangeRecord>,
    created_at: Instant,
    batch_id: String,
}

// ============================================================================
// Batch Processor
// ============================================================================

/// High-throughput batch processor for incremental updates
#[cfg(feature = "incremental")]
pub struct BatchProcessor {
    batch_size: usize,
    max_wait_time: Duration,
    pending_batches: DashMap<String, PendingBatch>,
    processing_semaphore: Semaphore,
    metrics: RwLock<BatchMetrics>,
}

#[cfg(feature = "incremental")]
impl BatchProcessor {
    /// Creates a new batch processor with specified configuration
    pub fn new(batch_size: usize, max_wait_time: Duration, max_concurrent_batches: usize) -> Self {
        Self {
            batch_size,
            max_wait_time,
            pending_batches: DashMap::new(),
            processing_semaphore: Semaphore::new(max_concurrent_batches),
            metrics: RwLock::new(BatchMetrics::default()),
        }
    }

    /// Adds a change to be processed in batches
    pub async fn add_change(&self, change: ChangeRecord) -> Result<String> {
        let batch_key = self.get_batch_key(&change);

        let batch_id = {
            let mut entry = self
                .pending_batches
                .entry(batch_key.clone())
                .or_insert_with(|| PendingBatch {
                    changes: Vec::new(),
                    created_at: Instant::now(),
                    batch_id: format!("batch_{}", Uuid::new_v4()),
                });

            entry.changes.push(change);
            let should_process = entry.changes.len() >= self.batch_size
                || entry.created_at.elapsed() > self.max_wait_time;

            let batch_id = entry.batch_id.clone();

            if should_process {
                // Move batch out for processing
                let batch = entry.clone();
                self.pending_batches.remove(&batch_key);

                // Process batch asynchronously
                let processor = Arc::new(self.clone());
                tokio::spawn(async move {
                    if let Err(e) = processor.process_batch(batch).await {
                        eprintln!("Batch processing error: {e}");
                    }
                });
            }

            batch_id
        };

        Ok(batch_id)
    }

    async fn process_batch(&self, batch: PendingBatch) -> Result<()> {
        let _permit = self.processing_semaphore.acquire().await.map_err(|_| {
            GraphRAGError::IncrementalUpdate {
                message: "Failed to acquire processing permit".to_string(),
            }
        })?;

        let start = Instant::now();

        // Group changes by type for optimized processing
        let mut entity_changes = Vec::new();
        let mut relationship_changes = Vec::new();
        let mut embedding_changes = Vec::new();

        for change in &batch.changes {
            match &change.change_type {
                ChangeType::EntityAdded | ChangeType::EntityUpdated | ChangeType::EntityRemoved => {
                    entity_changes.push(change);
                }
                ChangeType::RelationshipAdded
                | ChangeType::RelationshipUpdated
                | ChangeType::RelationshipRemoved => {
                    relationship_changes.push(change);
                }
                ChangeType::EmbeddingAdded
                | ChangeType::EmbeddingUpdated
                | ChangeType::EmbeddingRemoved => {
                    embedding_changes.push(change);
                }
                _ => {}
            }
        }

        // Process each type of change optimally
        self.process_entity_changes(&entity_changes).await?;
        self.process_relationship_changes(&relationship_changes)
            .await?;
        self.process_embedding_changes(&embedding_changes).await?;

        let processing_time = start.elapsed();

        // Update metrics
        self.update_metrics(&batch, processing_time).await;

        println!(
            "🚀 Processed batch {} with {} changes in {:?}",
            batch.batch_id,
            batch.changes.len(),
            processing_time
        );

        Ok(())
    }

    async fn process_entity_changes(&self, _changes: &[&ChangeRecord]) -> Result<()> {
        // Implementation would go here - process entity changes efficiently
        Ok(())
    }

    async fn process_relationship_changes(&self, _changes: &[&ChangeRecord]) -> Result<()> {
        // Implementation would go here - process relationship changes efficiently
        Ok(())
    }

    async fn process_embedding_changes(&self, _changes: &[&ChangeRecord]) -> Result<()> {
        // Implementation would go here - process embedding changes efficiently
        Ok(())
    }

    fn get_batch_key(&self, change: &ChangeRecord) -> String {
        // Group changes by entity or document for batching efficiency
        match (&change.entity_id, &change.document_id) {
            (Some(entity_id), _) => format!("entity:{entity_id}"),
            (None, Some(doc_id)) => format!("document:{doc_id}"),
            _ => "global".to_string(),
        }
    }

    async fn update_metrics(&self, batch: &PendingBatch, processing_time: Duration) {
        let mut metrics = self.metrics.write();

        metrics.total_batches_processed += 1;
        metrics.total_changes_processed += batch.changes.len() as u64;

        // Update running averages
        let total_batches = metrics.total_batches_processed as f64;
        metrics.average_batch_size = (metrics.average_batch_size * (total_batches - 1.0)
            + batch.changes.len() as f64)
            / total_batches;

        let prev_avg_ms = metrics.average_processing_time.as_millis() as f64;
        let new_avg_ms = (prev_avg_ms * (total_batches - 1.0) + processing_time.as_millis() as f64)
            / total_batches;
        metrics.average_processing_time = Duration::from_millis(new_avg_ms as u64);

        // Calculate throughput
        if processing_time.as_secs_f64() > 0.0 {
            metrics.throughput_per_second =
                batch.changes.len() as f64 / processing_time.as_secs_f64();
        }

        metrics.last_batch_processed = Some(Utc::now());
    }

    /// Gets the current batch processing metrics
    pub fn get_metrics(&self) -> BatchMetrics {
        self.metrics.read().clone()
    }
}

// Clone impl for BatchProcessor (required for Arc usage)
#[cfg(feature = "incremental")]
impl Clone for BatchProcessor {
    fn clone(&self) -> Self {
        Self {
            batch_size: self.batch_size,
            max_wait_time: self.max_wait_time,
            pending_batches: DashMap::new(), // New instance starts empty
            processing_semaphore: Semaphore::new(self.processing_semaphore.available_permits()),
            metrics: RwLock::new(self.get_metrics()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_metrics_default() {
        let metrics = BatchMetrics::default();
        assert_eq!(metrics.total_batches_processed, 0);
        assert_eq!(metrics.total_changes_processed, 0);
        assert_eq!(metrics.average_batch_size, 0.0);
    }

    #[cfg(feature = "incremental")]
    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new(100, Duration::from_millis(100), 10);
        assert_eq!(processor.batch_size, 100);
        
        let metrics = processor.get_metrics();
        assert_eq!(metrics.total_batches_processed, 0);
    }
}
