//! Community Report generation for Microsoft GraphRAG-compatible output
//!
//! This module provides LLM-generated community reports that fit entirely
//! in an LLM context window (~500-1000 tokens each).

use crate::core::{GraphRAGError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Phase 1: Data Model
// ============================================================================

/// A finding within a community report - a key insight about the community
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Finding {
    /// Short summary of the finding
    pub summary: String,
    /// Detailed explanation with data references like [Data: Entities (1, 2)]
    pub explanation: String,
}

/// Microsoft GraphRAG-compatible community report
///
/// Pre-computed LLM summary (~500-1000 tokens) designed to fit entirely
/// in an LLM coding assistant's context window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityReport {
    /// Unique identifier (format: "cr_{level}_{community_id}")
    pub id: String,
    /// Community identifier within its level
    pub community_id: usize,
    /// Hierarchical level (0 = finest granularity)
    pub level: usize,
    /// Short descriptive name for the community
    pub title: String,
    /// Executive summary (~100 tokens, 1-2 sentences)
    pub summary: String,
    /// Full report content (~500 tokens)
    pub full_content: String,
    /// Importance score (0.0-10.0, higher = more important)
    pub rank: f32,
    /// One sentence explanation of the rank
    pub rating_explanation: String,
    /// List of key insights (5-10 findings)
    pub findings: Vec<Finding>,
    /// Entity IDs belonging to this community
    pub entity_ids: Vec<String>,
    /// Number of entities in the community
    pub size: usize,
    /// Optional embedding of full_content for semantic search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

impl CommunityReport {
    /// Create a new community report
    pub fn new(community_id: usize, level: usize) -> Self {
        Self {
            id: format!("cr_{}_{}", level, community_id),
            community_id,
            level,
            title: String::new(),
            summary: String::new(),
            full_content: String::new(),
            rank: 0.0,
            rating_explanation: String::new(),
            findings: Vec::new(),
            entity_ids: Vec::new(),
            size: 0,
            embedding: None,
        }
    }

    /// Create from LLM response JSON
    pub fn from_llm_response(
        community_id: usize,
        level: usize,
        entity_ids: Vec<String>,
        response: &str,
    ) -> Result<Self> {
        // Parse JSON response from LLM
        let parsed: LLMReportResponse = serde_json::from_str(response).map_err(|e| {
            GraphRAGError::Generation {
                message: format!("Failed to parse LLM response as JSON: {e}"),
            }
        })?;

        let size = entity_ids.len();

        // Build full_content from parsed data
        let full_content = Self::build_full_content(&parsed);

        Ok(Self {
            id: format!("cr_{}_{}", level, community_id),
            community_id,
            level,
            title: parsed.title,
            summary: parsed.summary,
            full_content,
            rank: parsed.rating,
            rating_explanation: parsed.rating_explanation,
            findings: parsed
                .findings
                .into_iter()
                .map(|f| Finding {
                    summary: f.summary,
                    explanation: f.explanation,
                })
                .collect(),
            entity_ids,
            size,
            embedding: None,
        })
    }

    /// Build full_content string from parsed response
    fn build_full_content(parsed: &LLMReportResponse) -> String {
        let mut content = String::new();

        content.push_str(&format!("# {}\n\n", parsed.title));
        content.push_str(&format!("## Summary\n{}\n\n", parsed.summary));
        content.push_str(&format!(
            "## Importance: {:.1}/10\n{}\n\n",
            parsed.rating, parsed.rating_explanation
        ));

        if !parsed.findings.is_empty() {
            content.push_str("## Key Findings\n\n");
            for (i, finding) in parsed.findings.iter().enumerate() {
                content.push_str(&format!(
                    "### {}. {}\n{}\n\n",
                    i + 1,
                    finding.summary,
                    finding.explanation
                ));
            }
        }

        content
    }

    /// Get a compact representation suitable for LLM context
    /// Returns only title + summary (~100 tokens)
    pub fn compact(&self) -> String {
        format!("## {}\n{}", self.title, self.summary)
    }

    /// Get medium representation with findings summaries (~200 tokens)
    pub fn medium(&self) -> String {
        let findings_summary: String = self
            .findings
            .iter()
            .map(|f| format!("- {}", f.summary))
            .collect::<Vec<_>>()
            .join("\n");

        format!(
            "## {} (rank: {:.1})\n{}\n\nKey findings:\n{}",
            self.title, self.rank, self.summary, findings_summary
        )
    }
}

/// Internal struct for parsing LLM JSON response
#[derive(Debug, Deserialize)]
struct LLMReportResponse {
    title: String,
    summary: String,
    rating: f32,
    rating_explanation: String,
    findings: Vec<LLMFinding>,
}

#[derive(Debug, Deserialize)]
struct LLMFinding {
    summary: String,
    explanation: String,
}

// ============================================================================
// Phase 2: Prompt Template
// ============================================================================

/// Prompt template for generating community reports (adapted from Microsoft GraphRAG)
pub const COMMUNITY_REPORT_PROMPT: &str = r#"You are an AI assistant analyzing a community of related entities in a codebase or knowledge graph.

# Goal
Write a comprehensive report about this community of entities, identifying key patterns, relationships, and their significance.

# Report Structure
Return a JSON object with the following structure:
{
    "title": "<short descriptive name that represents key entities - be specific>",
    "summary": "<1-2 sentence executive summary of the community's structure and significance>",
    "rating": <float 0-10 representing importance/impact>,
    "rating_explanation": "<one sentence explaining the rating>",
    "findings": [
        {
            "summary": "<short insight title>",
            "explanation": "<detailed explanation with data references like [Data: Entities (id1, id2)]>"
        }
    ]
}

# Grounding Rules
- All claims must reference supporting data: [Data: Entities (ids)] or [Data: Relationships (ids)]
- Do not list more than 5 IDs per reference; use "+more" for additional items
- Do not include unsupported information
- Provide 3-7 findings depending on community complexity

# Community Entities
{entities}

# Community Relationships
{relationships}

# Additional Context
{context}

Return ONLY valid JSON, no markdown code blocks or additional text."#;

/// Prompt template for code-specific community reports
pub const CODE_COMMUNITY_REPORT_PROMPT: &str = r#"You are an AI assistant analyzing a community of related code entities (functions, structs, modules, traits) in a Rust codebase.

# Goal
Write a technical report about this code community, identifying architectural patterns, dependencies, and their significance for developers.

# Report Structure
Return a JSON object:
{
    "title": "<descriptive name for this code subsystem>",
    "summary": "<1-2 sentence summary of what this code does and why it matters>",
    "rating": <float 0-10: 10=critical infrastructure, 5=utility code, 0=dead code>,
    "rating_explanation": "<why this rating>",
    "findings": [
        {
            "summary": "<pattern or insight>",
            "explanation": "<technical detail with [Data: Entities (ids)]>"
        }
    ]
}

# Code Entities
{entities}

# Call Relationships
{relationships}

# Module Context
{context}

Return ONLY valid JSON."#;

// ============================================================================
// Phase 3: Configuration
// ============================================================================

/// Configuration for community report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunityReportConfig {
    /// Maximum tokens for LLM response
    pub max_tokens: usize,
    /// Temperature for LLM generation (0.0-1.0)
    pub temperature: f32,
    /// Minimum community size to generate report for
    pub min_community_size: usize,
    /// Maximum findings per report
    pub max_findings: usize,
    /// Which levels to generate reports for (empty = all)
    pub levels: Vec<usize>,
    /// Use code-specific prompt template
    pub use_code_prompt: bool,
    /// Include entity descriptions in prompt
    pub include_descriptions: bool,
    /// Maximum entities to include in prompt (for token limits)
    pub max_entities_in_prompt: usize,
    /// Maximum relationships to include in prompt
    pub max_relationships_in_prompt: usize,
}

impl Default for CommunityReportConfig {
    fn default() -> Self {
        Self {
            max_tokens: 2000,
            temperature: 0.3, // Lower temp for more consistent structured output
            min_community_size: 2,
            max_findings: 7,
            levels: vec![], // All levels
            use_code_prompt: false,
            include_descriptions: true,
            max_entities_in_prompt: 50,
            max_relationships_in_prompt: 100,
        }
    }
}

impl CommunityReportConfig {
    /// Create config optimized for code analysis
    pub fn for_code() -> Self {
        Self {
            use_code_prompt: true,
            max_findings: 5,
            ..Default::default()
        }
    }

    /// Create config for large communities (token-conscious)
    pub fn for_large_communities() -> Self {
        Self {
            max_entities_in_prompt: 30,
            max_relationships_in_prompt: 50,
            include_descriptions: false,
            ..Default::default()
        }
    }
}

// ============================================================================
// Phase 3: Generator Implementation
// ============================================================================

use crate::summarization::LLMClient;

/// Input data for generating a single community report
#[derive(Debug, Clone)]
pub struct CommunityInput {
    /// Community ID
    pub community_id: usize,
    /// Hierarchical level
    pub level: usize,
    /// Entities: (id, name, type_or_description)
    pub entities: Vec<(String, String, String)>,
    /// Relationships: (source_id, target_id, relation_type)
    pub relationships: Vec<(String, String, String)>,
    /// Additional context (e.g., module path, parent community summary)
    pub context: String,
}

/// Generator for community reports using LLM
pub struct CommunityReportGenerator {
    config: CommunityReportConfig,
}

impl CommunityReportGenerator {
    /// Create a new generator with configuration
    pub fn new(config: CommunityReportConfig) -> Self {
        Self { config }
    }

    /// Get the prompt template based on config
    pub fn get_prompt_template(&self) -> &'static str {
        if self.config.use_code_prompt {
            CODE_COMMUNITY_REPORT_PROMPT
        } else {
            COMMUNITY_REPORT_PROMPT
        }
    }

    /// Format entities for prompt
    pub fn format_entities(&self, entities: &[(String, String, String)]) -> String {
        // entities: Vec<(id, name, type)> or Vec<(id, name, description)>
        let limit = self.config.max_entities_in_prompt.min(entities.len());
        let mut result = String::new();

        for (id, name, extra) in entities.iter().take(limit) {
            if self.config.include_descriptions && !extra.is_empty() {
                result.push_str(&format!("- {id}: {name} ({extra})\n"));
            } else {
                result.push_str(&format!("- {id}: {name}\n"));
            }
        }

        if entities.len() > limit {
            result.push_str(&format!("... and {} more entities\n", entities.len() - limit));
        }

        result
    }

    /// Format relationships for prompt
    pub fn format_relationships(&self, relationships: &[(String, String, String)]) -> String {
        // relationships: Vec<(source_id, target_id, relation_type)>
        let limit = self.config.max_relationships_in_prompt.min(relationships.len());
        let mut result = String::new();

        for (source, target, rel_type) in relationships.iter().take(limit) {
            result.push_str(&format!("- {source} --[{rel_type}]--> {target}\n"));
        }

        if relationships.len() > limit {
            result.push_str(&format!(
                "... and {} more relationships\n",
                relationships.len() - limit
            ));
        }

        result
    }

    /// Build the full prompt for a community
    pub fn build_prompt(
        &self,
        entities: &[(String, String, String)],
        relationships: &[(String, String, String)],
        context: &str,
    ) -> String {
        let template = self.get_prompt_template();
        let entities_str = self.format_entities(entities);
        let relationships_str = self.format_relationships(relationships);

        template
            .replace("{entities}", &entities_str)
            .replace("{relationships}", &relationships_str)
            .replace("{context}", context)
    }

    /// Get config reference
    pub fn config(&self) -> &CommunityReportConfig {
        &self.config
    }

    /// Generate a single community report using LLM
    #[cfg(feature = "async")]
    pub async fn generate_report<L: LLMClient>(
        &self,
        input: &CommunityInput,
        llm: &L,
    ) -> Result<CommunityReport> {
        // Skip communities below minimum size
        if input.entities.len() < self.config.min_community_size {
            return Err(GraphRAGError::Generation {
                message: format!(
                    "Community {} has {} entities, below minimum {}",
                    input.community_id,
                    input.entities.len(),
                    self.config.min_community_size
                ),
            });
        }

        // Build prompt
        let prompt = self.build_prompt(&input.entities, &input.relationships, &input.context);

        // Call LLM
        let response = llm
            .generate_summary("", &prompt, self.config.max_tokens, self.config.temperature)
            .await?;

        // Extract entity IDs
        let entity_ids: Vec<String> = input.entities.iter().map(|(id, _, _)| id.clone()).collect();

        // Parse response into CommunityReport
        CommunityReport::from_llm_response(input.community_id, input.level, entity_ids, &response)
    }

    /// Generate reports for multiple communities
    #[cfg(feature = "async")]
    pub async fn generate_reports<L: LLMClient>(
        &self,
        inputs: &[CommunityInput],
        llm: &L,
    ) -> Result<CommunityReports> {
        let mut reports = CommunityReports::new();
        let mut errors = Vec::new();

        for input in inputs {
            // Check level filter
            if !self.config.levels.is_empty() && !self.config.levels.contains(&input.level) {
                continue;
            }

            match self.generate_report(input, llm).await {
                Ok(report) => {
                    tracing::debug!(
                        "Generated report for community {} at level {}: {}",
                        input.community_id,
                        input.level,
                        report.title
                    );
                    reports.add(report);
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to generate report for community {} at level {}: {}",
                        input.community_id,
                        input.level,
                        e
                    );
                    errors.push((input.community_id, input.level, e.to_string()));
                }
            }
        }

        if reports.is_empty() && !errors.is_empty() {
            return Err(GraphRAGError::Generation {
                message: format!("All {} report generations failed", errors.len()),
            });
        }

        tracing::info!(
            "Generated {} community reports ({} failures)",
            reports.len(),
            errors.len()
        );

        Ok(reports)
    }

    /// Generate reports with batch LLM calls for efficiency
    #[cfg(feature = "async")]
    pub async fn generate_reports_batch<L: LLMClient>(
        &self,
        inputs: &[CommunityInput],
        llm: &L,
        batch_size: usize,
    ) -> Result<CommunityReports> {
        let mut reports = CommunityReports::new();

        // Filter inputs by level and size
        let filtered_inputs: Vec<_> = inputs
            .iter()
            .filter(|input| {
                let level_ok =
                    self.config.levels.is_empty() || self.config.levels.contains(&input.level);
                let size_ok = input.entities.len() >= self.config.min_community_size;
                level_ok && size_ok
            })
            .collect();

        // Process in batches
        for chunk in filtered_inputs.chunks(batch_size) {
            // Build prompts for batch
            let batch_data: Vec<_> = chunk
                .iter()
                .map(|input| {
                    let prompt =
                        self.build_prompt(&input.entities, &input.relationships, &input.context);
                    let entity_ids: Vec<String> =
                        input.entities.iter().map(|(id, _, _)| id.clone()).collect();
                    (input.community_id, input.level, entity_ids, prompt)
                })
                .collect();

            // Call LLM in batch
            let prompts: Vec<(&str, &str)> = batch_data
                .iter()
                .map(|(_, _, _, prompt)| ("", prompt.as_str()))
                .collect();

            match llm
                .generate_summary_batch(&prompts, self.config.max_tokens, self.config.temperature)
                .await
            {
                Ok(responses) => {
                    for ((community_id, level, entity_ids, _), response) in
                        batch_data.into_iter().zip(responses)
                    {
                        match CommunityReport::from_llm_response(
                            community_id,
                            level,
                            entity_ids,
                            &response,
                        ) {
                            Ok(report) => reports.add(report),
                            Err(e) => {
                                tracing::warn!(
                                    "Failed to parse report for community {}: {}",
                                    community_id,
                                    e
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Batch LLM call failed: {}", e);
                }
            }
        }

        Ok(reports)
    }

    /// Generate an extractive (non-LLM) report as fallback
    pub fn generate_extractive_report(&self, input: &CommunityInput) -> CommunityReport {
        let entity_ids: Vec<String> = input.entities.iter().map(|(id, _, _)| id.clone()).collect();
        let size = entity_ids.len();

        // Build title from top entities
        let title = if input.entities.len() <= 3 {
            input
                .entities
                .iter()
                .map(|(_, name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        } else {
            format!(
                "{}, {}, and {} others",
                input.entities[0].1,
                input.entities[1].1,
                input.entities.len() - 2
            )
        };

        // Build summary
        let entity_types: HashMap<String, usize> =
            input
                .entities
                .iter()
                .fold(HashMap::new(), |mut acc, (_, _, t)| {
                    *acc.entry(t.clone()).or_insert(0) += 1;
                    acc
                });
        let type_summary: String = entity_types
            .iter()
            .map(|(t, c)| format!("{} {}", c, t))
            .collect::<Vec<_>>()
            .join(", ");

        let summary = format!(
            "Community of {} entities at level {}: {}.",
            size, input.level, type_summary
        );

        // Build findings from relationships
        let mut findings = Vec::new();
        if !input.relationships.is_empty() {
            let rel_count = input.relationships.len();
            findings.push(Finding {
                summary: format!("{} relationships identified", rel_count),
                explanation: format!(
                    "The community has {} internal relationships connecting its {} entities.",
                    rel_count, size
                ),
            });
        }

        // Build full content
        let mut full_content = format!("# {}\n\n## Summary\n{}\n\n", title, summary);
        full_content.push_str("## Entities\n");
        for (id, name, etype) in &input.entities {
            full_content.push_str(&format!("- {} ({}) [{}]\n", name, etype, id));
        }

        CommunityReport {
            id: format!("cr_{}_{}", input.level, input.community_id),
            community_id: input.community_id,
            level: input.level,
            title,
            summary,
            full_content,
            rank: (size as f32 / 10.0).min(10.0), // Simple rank based on size
            rating_explanation: format!("Rank based on community size ({} entities)", size),
            findings,
            entity_ids,
            size,
            embedding: None,
        }
    }
}

// ============================================================================
// Collection type for multiple reports
// ============================================================================

/// Collection of community reports with helper methods
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommunityReports {
    /// All generated reports
    pub reports: Vec<CommunityReport>,
    /// Index by level for fast lookup
    #[serde(skip)]
    level_index: HashMap<usize, Vec<usize>>,
}

impl CommunityReports {
    /// Create empty collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from vector of reports
    pub fn from_reports(reports: Vec<CommunityReport>) -> Self {
        let mut collection = Self {
            reports,
            level_index: HashMap::new(),
        };
        collection.rebuild_index();
        collection
    }

    /// Rebuild the level index
    fn rebuild_index(&mut self) {
        self.level_index.clear();
        for (idx, report) in self.reports.iter().enumerate() {
            self.level_index
                .entry(report.level)
                .or_default()
                .push(idx);
        }
    }

    /// Add a report
    pub fn add(&mut self, report: CommunityReport) {
        let level = report.level;
        let idx = self.reports.len();
        self.reports.push(report);
        self.level_index.entry(level).or_default().push(idx);
    }

    /// Get reports at a specific level
    pub fn at_level(&self, level: usize) -> Vec<&CommunityReport> {
        self.level_index
            .get(&level)
            .map(|indices| indices.iter().map(|&i| &self.reports[i]).collect())
            .unwrap_or_default()
    }

    /// Get all available levels
    pub fn levels(&self) -> Vec<usize> {
        let mut levels: Vec<_> = self.level_index.keys().copied().collect();
        levels.sort();
        levels
    }

    /// Get top N reports by rank
    pub fn top_by_rank(&self, n: usize) -> Vec<&CommunityReport> {
        let mut sorted: Vec<_> = self.reports.iter().collect();
        sorted.sort_by(|a, b| b.rank.partial_cmp(&a.rank).unwrap_or(std::cmp::Ordering::Equal));
        sorted.into_iter().take(n).collect()
    }

    /// Get compact summaries for all reports at a level (fits in context window)
    pub fn compact_summaries_at_level(&self, level: usize) -> String {
        self.at_level(level)
            .iter()
            .map(|r| r.compact())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get total token estimate for reports at level
    pub fn estimated_tokens_at_level(&self, level: usize) -> usize {
        // Rough estimate: 4 chars per token
        self.at_level(level)
            .iter()
            .map(|r| r.full_content.len() / 4)
            .sum()
    }

    /// Find reports containing a specific entity
    pub fn with_entity(&self, entity_id: &str) -> Vec<&CommunityReport> {
        self.reports
            .iter()
            .filter(|r| r.entity_ids.iter().any(|id| id == entity_id))
            .collect()
    }

    /// Total report count
    pub fn len(&self) -> usize {
        self.reports.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.reports.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_community_report_new() {
        let report = CommunityReport::new(5, 0);
        assert_eq!(report.id, "cr_0_5");
        assert_eq!(report.community_id, 5);
        assert_eq!(report.level, 0);
    }

    #[test]
    fn test_community_report_from_llm_response() {
        let json = r#"{
            "title": "MCP Server Components",
            "summary": "Core server infrastructure for Model Context Protocol.",
            "rating": 8.5,
            "rating_explanation": "Critical infrastructure component.",
            "findings": [
                {
                    "summary": "Central hub pattern",
                    "explanation": "Server acts as central hub [Data: Entities (1, 2, 3)]"
                }
            ]
        }"#;

        let entity_ids = vec!["e1".to_string(), "e2".to_string(), "e3".to_string()];
        let report = CommunityReport::from_llm_response(5, 0, entity_ids, json).unwrap();

        assert_eq!(report.title, "MCP Server Components");
        assert_eq!(report.rank, 8.5);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.size, 3);
    }

    #[test]
    fn test_community_report_compact() {
        let mut report = CommunityReport::new(1, 0);
        report.title = "Test Community".to_string();
        report.summary = "A test summary.".to_string();

        let compact = report.compact();
        assert!(compact.contains("Test Community"));
        assert!(compact.contains("A test summary."));
    }

    #[test]
    fn test_generator_format_entities() {
        let config = CommunityReportConfig::default();
        let generator = CommunityReportGenerator::new(config);

        let entities = vec![
            ("e1".to_string(), "Entity1".to_string(), "Type A".to_string()),
            ("e2".to_string(), "Entity2".to_string(), "Type B".to_string()),
        ];

        let formatted = generator.format_entities(&entities);
        assert!(formatted.contains("e1: Entity1"));
        assert!(formatted.contains("e2: Entity2"));
    }

    #[test]
    fn test_community_reports_collection() {
        let mut collection = CommunityReports::new();

        let mut r1 = CommunityReport::new(1, 0);
        r1.rank = 8.0;
        r1.entity_ids = vec!["e1".to_string()];

        let mut r2 = CommunityReport::new(2, 0);
        r2.rank = 5.0;
        r2.entity_ids = vec!["e2".to_string()];

        let mut r3 = CommunityReport::new(1, 1);
        r3.rank = 7.0;
        r3.entity_ids = vec!["e1".to_string(), "e2".to_string()];

        collection.add(r1);
        collection.add(r2);
        collection.add(r3);

        assert_eq!(collection.len(), 3);
        assert_eq!(collection.at_level(0).len(), 2);
        assert_eq!(collection.at_level(1).len(), 1);
        assert_eq!(collection.top_by_rank(2).len(), 2);
        assert_eq!(collection.with_entity("e1").len(), 2);
    }
}
