//! Change records and graph deltas for tracking modifications.
//!
//! This module contains types for recording individual changes and
//! grouping them into atomic delta operations.

use crate::core::{DocumentId, Entity, EntityId, Relationship, TextChunk};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{ChangeType, DeltaStatus, Operation, UpdateId};

// ============================================================================
// Change Records
// ============================================================================

/// Change record for tracking individual modifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeRecord {
    /// Unique identifier for this change
    pub change_id: UpdateId,
    /// Timestamp when the change occurred
    pub timestamp: DateTime<Utc>,
    /// Type of change performed
    pub change_type: ChangeType,
    /// Optional entity ID affected by this change
    pub entity_id: Option<EntityId>,
    /// Optional document ID affected by this change
    pub document_id: Option<DocumentId>,
    /// Operation type (insert, update, delete, upsert)
    pub operation: Operation,
    /// Data associated with the change
    pub data: ChangeData,
    /// Additional metadata for the change
    pub metadata: HashMap<String, String>,
}

/// Data associated with a change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeData {
    /// Entity data
    Entity(Entity),
    /// Relationship data
    Relationship(Relationship),
    /// Document data
    Document(Document),
    /// Text chunk data
    Chunk(TextChunk),
    /// Embedding data with entity ID and vector
    Embedding {
        /// Entity ID for the embedding
        entity_id: EntityId,
        /// Embedding vector
        embedding: Vec<f32>,
    },
    /// Empty change data placeholder
    Empty,
}

impl ChangeData {
    /// Extracts the entity ID from this change data if applicable
    pub fn get_entity_id(&self) -> Option<EntityId> {
        match self {
            ChangeData::Entity(entity) => Some(entity.id.clone()),
            ChangeData::Embedding { entity_id, .. } => Some(entity_id.clone()),
            _ => None,
        }
    }
}

// ============================================================================
// Document Type
// ============================================================================

/// Document type for incremental updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique identifier for the document
    pub id: DocumentId,
    /// Document title
    pub title: String,
    /// Document content
    pub content: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

// ============================================================================
// Graph Delta
// ============================================================================

/// Atomic change set representing a transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDelta {
    /// Unique identifier for this delta
    pub delta_id: UpdateId,
    /// Timestamp when the delta was created
    pub timestamp: DateTime<Utc>,
    /// List of changes in this delta
    pub changes: Vec<ChangeRecord>,
    /// Delta IDs that this delta depends on
    pub dependencies: Vec<UpdateId>,
    /// Current status of the delta
    pub status: DeltaStatus,
    /// Data needed to rollback this delta
    pub rollback_data: Option<RollbackData>,
}

/// Data needed for rollback operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackData {
    /// Previous state of entities before the change
    pub previous_entities: Vec<Entity>,
    /// Previous state of relationships before the change
    pub previous_relationships: Vec<Relationship>,
    /// Cache keys affected by the change
    pub affected_caches: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Entity;
    use crate::graph::incremental::types::Operation;

    #[test]
    fn test_change_record_creation() {
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test Entity".to_string(),
            "Person".to_string(),
            0.9,
        );

        let change = ChangeRecord {
            change_id: UpdateId::new(),
            timestamp: Utc::now(),
            change_type: ChangeType::EntityAdded,
            entity_id: Some(entity.id.clone()),
            document_id: None,
            operation: Operation::Insert,
            data: ChangeData::Entity(entity),
            metadata: HashMap::new(),
        };

        assert!(matches!(change.change_type, ChangeType::EntityAdded));
        assert!(matches!(change.operation, Operation::Insert));
    }

    #[test]
    fn test_change_data_get_entity_id() {
        let entity = Entity::new(
            EntityId::new("test".to_string()),
            "Test".to_string(),
            "Person".to_string(),
            0.9,
        );
        let data = ChangeData::Entity(entity.clone());
        assert_eq!(data.get_entity_id(), Some(entity.id));

        let empty = ChangeData::Empty;
        assert_eq!(empty.get_entity_id(), None);
    }
}
