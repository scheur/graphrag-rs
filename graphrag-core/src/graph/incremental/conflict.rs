//! Conflict detection and resolution for concurrent graph modifications.
//!
//! This module provides strategies and mechanisms for resolving conflicts
//! when multiple updates attempt to modify the same graph elements.

use crate::core::{Entity, GraphRAGError, Relationship, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::change_log::ChangeData;
use super::types::UpdateId;

// ============================================================================
// Conflict Types
// ============================================================================

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictStrategy {
    /// Keep the existing data, discard new changes
    KeepExisting,
    /// Keep the new data, discard existing
    KeepNew,
    /// Merge existing and new data intelligently
    Merge,
    /// Use LLM to decide how to resolve conflict
    LLMDecision,
    /// Prompt user to resolve conflict
    UserPrompt,
    /// Use a custom resolver by name
    Custom(String),
}

/// Conflict detected during update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// Unique identifier for this conflict
    pub conflict_id: UpdateId,
    /// Type of conflict detected
    pub conflict_type: ConflictType,
    /// Existing data in the graph
    pub existing_data: ChangeData,
    /// New data attempting to be applied
    pub new_data: ChangeData,
    /// Resolution if already resolved
    pub resolution: Option<ConflictResolution>,
}

/// Types of conflicts that can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    /// Entity already exists with different data
    EntityExists,
    /// Relationship already exists with different data
    RelationshipExists,
    /// Version mismatch between expected and actual
    VersionMismatch,
    /// Data is inconsistent with graph state
    DataInconsistency,
    /// Change violates a constraint
    ConstraintViolation,
}

/// Resolution for a conflict
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolution {
    /// Strategy used to resolve the conflict
    pub strategy: ConflictStrategy,
    /// Resolved data after applying strategy
    pub resolved_data: ChangeData,
    /// Metadata about the resolution
    pub metadata: HashMap<String, String>,
}

// ============================================================================
// Conflict Resolver
// ============================================================================

/// Type alias for custom resolver functions
type ConflictResolverFn = Box<dyn Fn(&Conflict) -> Result<ConflictResolution> + Send + Sync>;

/// Conflict resolver with multiple strategies
pub struct ConflictResolver {
    strategy: ConflictStrategy,
    custom_resolvers: HashMap<String, ConflictResolverFn>,
}

impl ConflictResolver {
    /// Creates a new conflict resolver with the given strategy
    pub fn new(strategy: ConflictStrategy) -> Self {
        Self {
            strategy,
            custom_resolvers: HashMap::new(),
        }
    }

    /// Adds a custom resolver function by name
    pub fn with_custom_resolver<F>(mut self, name: String, resolver: F) -> Self
    where
        F: Fn(&Conflict) -> Result<ConflictResolution> + Send + Sync + 'static,
    {
        self.custom_resolvers.insert(name, Box::new(resolver));
        self
    }

    /// Resolves a conflict using the configured strategy
    pub async fn resolve_conflict(&self, conflict: &Conflict) -> Result<ConflictResolution> {
        self.resolve_conflict_with_strategy(conflict, &self.strategy)
            .await
    }

    /// Resolves a conflict using a specific strategy (overrides configured strategy)
    pub async fn resolve_conflict_with_strategy(
        &self,
        conflict: &Conflict,
        strategy: &ConflictStrategy,
    ) -> Result<ConflictResolution> {
        match strategy {
            ConflictStrategy::KeepExisting => Ok(ConflictResolution {
                strategy: ConflictStrategy::KeepExisting,
                resolved_data: conflict.existing_data.clone(),
                metadata: HashMap::new(),
            }),
            ConflictStrategy::KeepNew => Ok(ConflictResolution {
                strategy: ConflictStrategy::KeepNew,
                resolved_data: conflict.new_data.clone(),
                metadata: HashMap::new(),
            }),
            ConflictStrategy::Merge => self.merge_conflict_data(conflict).await,
            ConflictStrategy::Custom(resolver_name) => {
                if let Some(resolver) = self.custom_resolvers.get(resolver_name) {
                    resolver(conflict)
                } else {
                    Err(GraphRAGError::ConflictResolution {
                        message: format!("Custom resolver '{resolver_name}' not found"),
                    })
                }
            }
            _ => Err(GraphRAGError::ConflictResolution {
                message: "Conflict resolution strategy not implemented".to_string(),
            }),
        }
    }

    async fn merge_conflict_data(&self, conflict: &Conflict) -> Result<ConflictResolution> {
        match (&conflict.existing_data, &conflict.new_data) {
            (ChangeData::Entity(existing), ChangeData::Entity(new)) => {
                let merged = self.merge_entities(existing, new)?;
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Merge,
                    resolved_data: ChangeData::Entity(merged),
                    metadata: [("merge_strategy".to_string(), "entity_merge".to_string())]
                        .into_iter()
                        .collect(),
                })
            }
            (ChangeData::Relationship(existing), ChangeData::Relationship(new)) => {
                let merged = self.merge_relationships(existing, new)?;
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Merge,
                    resolved_data: ChangeData::Relationship(merged),
                    metadata: [(
                        "merge_strategy".to_string(),
                        "relationship_merge".to_string(),
                    )]
                    .into_iter()
                    .collect(),
                })
            }
            _ => Err(GraphRAGError::ConflictResolution {
                message: "Cannot merge incompatible data types".to_string(),
            }),
        }
    }

    fn merge_entities(&self, existing: &Entity, new: &Entity) -> Result<Entity> {
        let mut merged = existing.clone();

        // Use higher confidence
        if new.confidence > existing.confidence {
            merged.confidence = new.confidence;
            merged.name = new.name.clone();
            merged.entity_type = new.entity_type.clone();
        }

        // Merge mentions
        let mut all_mentions = existing.mentions.clone();
        for new_mention in &new.mentions {
            if !all_mentions.iter().any(|m| {
                m.chunk_id == new_mention.chunk_id && m.start_offset == new_mention.start_offset
            }) {
                all_mentions.push(new_mention.clone());
            }
        }
        merged.mentions = all_mentions;

        // Prefer new embedding if available
        if new.embedding.is_some() {
            merged.embedding = new.embedding.clone();
        }

        Ok(merged)
    }

    fn merge_relationships(
        &self,
        existing: &Relationship,
        new: &Relationship,
    ) -> Result<Relationship> {
        let mut merged = existing.clone();

        // Use higher confidence
        if new.confidence > existing.confidence {
            merged.confidence = new.confidence;
            merged.relation_type = new.relation_type.clone();
        }

        // Merge contexts
        let mut all_contexts = existing.context.clone();
        for new_context in &new.context {
            if !all_contexts.contains(new_context) {
                all_contexts.push(new_context.clone());
            }
        }
        merged.context = all_contexts;

        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conflict_resolver_creation() {
        let resolver = ConflictResolver::new(ConflictStrategy::Merge);
        assert!(matches!(resolver.strategy, ConflictStrategy::Merge));
    }

    #[test]
    fn test_conflict_resolver_with_custom() {
        let resolver = ConflictResolver::new(ConflictStrategy::Merge).with_custom_resolver(
            "test".to_string(),
            |conflict| {
                Ok(ConflictResolution {
                    strategy: ConflictStrategy::Custom("test".to_string()),
                    resolved_data: conflict.new_data.clone(),
                    metadata: HashMap::new(),
                })
            },
        );
        assert!(resolver.custom_resolvers.contains_key("test"));
    }
}
