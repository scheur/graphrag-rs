///! Analyze a Rust codebase with GraphRAG + Candle (Metal) embeddings.
///! Uses CODE-AWARE entity extraction (FUNCTION, STRUCT, TRAIT, etc.)
///!
///! Usage (from crates/graphrag-rs):
///!   cargo run -p graphrag-core --example cargo_xray_code --features metal -- ../..
use graphrag_core::core::error::{GraphRAGError, Result};
use graphrag_core::embeddings::{neural::CandleEmbedder, EmbeddingProvider};
use graphrag_core::nlp::custom_ner::{CustomNER, EntityType, ExtractionRule, RuleType};
use graphrag_core::{Config, GraphRAG};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use tracing_subscriber::EnvFilter;

const MAX_FILE_BYTES: u64 = 512 * 1024;
const MAX_FILES: usize = 800;
const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Build a CustomNER configured for Rust code entities
fn build_rust_ner() -> CustomNER {
    let mut ner = CustomNER::new();

    // Register code-aware entity types
    let entity_types = vec![
        ("FUNCTION", "Rust function definitions"),
        ("STRUCT", "Rust struct definitions"),
        ("ENUM", "Rust enum definitions"),
        ("TRAIT", "Rust trait definitions"),
        ("IMPL", "Rust impl blocks"),
        ("MODULE", "Rust module definitions"),
        ("CRATE", "External crate references"),
        ("TYPE", "Type alias definitions"),
        ("CONST", "Constant definitions"),
        ("MACRO", "Macro definitions"),
    ];

    for (name, desc) in entity_types {
        ner.register_entity_type(EntityType::new(name.to_string(), desc.to_string()));
    }

    // Add extraction rules with regex patterns
    let rules = vec![
        // Functions: fn name(
        ExtractionRule {
            name: "rust_function".to_string(),
            entity_type: "FUNCTION".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)\s*[<(]".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Structs: struct Name
        ExtractionRule {
            name: "rust_struct".to_string(),
            entity_type: "STRUCT".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9]*)".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Enums: enum Name
        ExtractionRule {
            name: "rust_enum".to_string(),
            entity_type: "ENUM".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9]*)".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Traits: trait Name
        ExtractionRule {
            name: "rust_trait".to_string(),
            entity_type: "TRAIT".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9]*)".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Impl blocks: impl Type or impl Trait for Type
        ExtractionRule {
            name: "rust_impl".to_string(),
            entity_type: "IMPL".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"impl(?:<[^>]+>)?\s+(?:([A-Z][a-zA-Z0-9]*)\s+for\s+)?([A-Z][a-zA-Z0-9]*)".to_string(),
            min_confidence: 0.90,
            priority: 8,
        },
        // Modules: mod name
        ExtractionRule {
            name: "rust_module".to_string(),
            entity_type: "MODULE".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?mod\s+([a-z_][a-z0-9_]*)".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Crate references: use crate_name::
        ExtractionRule {
            name: "rust_crate".to_string(),
            entity_type: "CRATE".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"use\s+([a-z_][a-z0-9_]*)::".to_string(),
            min_confidence: 0.85,
            priority: 5,
        },
        // Type aliases: type Name =
        ExtractionRule {
            name: "rust_type".to_string(),
            entity_type: "TYPE".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?type\s+([A-Z][a-zA-Z0-9]*)\s*(?:<[^>]+>)?\s*=".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Constants: const NAME
        ExtractionRule {
            name: "rust_const".to_string(),
            entity_type: "CONST".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"(?:pub\s+)?const\s+([A-Z_][A-Z0-9_]*)\s*:".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
        // Macros: macro_rules! name
        ExtractionRule {
            name: "rust_macro".to_string(),
            entity_type: "MACRO".to_string(),
            rule_type: RuleType::Regex,
            pattern: r"macro_rules!\s+([a-z_][a-z0-9_]*)".to_string(),
            min_confidence: 0.95,
            priority: 10,
        },
    ];

    for rule in rules {
        ner.add_rule(rule);
    }

    ner
}

fn should_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git" | "target" | "node_modules" | "dist" | "build" | ".next" | ".cache" | "vendor" | "tmp"
    )
}

fn should_include_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("rs") | Some("toml") | Some("md") => true,
        _ => false,
    }
}

fn collect_files(root: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root.to_path_buf());

    while let Some(dir) = queue.pop_front() {
        let entries = match fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };
            if file_type.is_symlink() {
                continue;
            }
            let path = entry.path();
            if file_type.is_dir() {
                if name.starts_with('.') || should_skip_dir(&name) {
                    continue;
                }
                queue.push_back(path);
                continue;
            }
            if !file_type.is_file() || !should_include_file(&path) {
                continue;
            }
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if metadata.len() > MAX_FILE_BYTES {
                continue;
            }
            files.push(path);
            if files.len() >= MAX_FILES {
                return Ok(files);
            }
        }
    }

    Ok(files)
}

fn read_file_content(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    let text = String::from_utf8(bytes).ok()?;
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("graphrag_core=info"))
        .init();

    let mut args = std::env::args().skip(1);
    let root_arg = args.next().unwrap_or_else(|| ".".to_string());
    let output_arg = args
        .next()
        .unwrap_or_else(|| "./graphrag-output-code".to_string());

    let root = PathBuf::from(root_arg);
    let root = root.canonicalize().unwrap_or(root);
    if !root.is_dir() {
        return Err(GraphRAGError::Config {
            message: format!("Root path is not a directory: {}", root.display()),
        });
    }

    let output_dir = PathBuf::from(output_arg);
    fs::create_dir_all(&output_dir)?;

    println!("📂 Scanning {} for Rust code...", root.display());
    let files = collect_files(&root)?;
    println!("✅ Collected {} files (max {})", files.len(), MAX_FILES);

    // Configure for CODE entity types
    let mut config = Config::default();
    config.output_dir = output_dir.display().to_string();
    config.embeddings.backend = "candle".to_string();
    config.embeddings.model = Some(MODEL_ID.to_string());
    config.embeddings.fallback_to_hash = false;
    config.approach = "algorithmic".to_string();
    config.entities.use_gleaning = false;
    // CODE-AWARE entity types
    config.entities.entity_types = vec![
        "FUNCTION".to_string(),
        "STRUCT".to_string(),
        "ENUM".to_string(),
        "TRAIT".to_string(),
        "IMPL".to_string(),
        "MODULE".to_string(),
        "CRATE".to_string(),
        "TYPE".to_string(),
        "CONST".to_string(),
        "MACRO".to_string(),
    ];
    config.graph.extract_relationships = true;

    let mut graphrag = GraphRAG::new(config)?;
    graphrag.initialize()?;

    // Build custom NER for Rust code
    let rust_ner = build_rust_ner();
    let mut all_entities: Vec<(String, String, usize)> = Vec::new(); // (name, type, count)

    println!("🔍 Extracting code entities...");
    let total_files = files.len();
    for (idx, path) in files.iter().enumerate() {
        if let Some(content) = read_file_content(path) {
            // Extract entities using our Rust NER
            let entities = rust_ner.extract(&content);

            // Count unique entities per file
            for entity in &entities {
                if let Some(pos) = all_entities
                    .iter()
                    .position(|(n, t, _)| n == &entity.text && t == &entity.entity_type)
                {
                    all_entities[pos].2 += 1;
                } else {
                    all_entities.push((entity.text.clone(), entity.entity_type.clone(), 1));
                }
            }

            // Add document with file path prefix
            let decorated = format!("// file: {}\n{}", path.display(), content);
            graphrag.add_document_from_text(&decorated)?;

            if (idx + 1) % 50 == 0 || idx + 1 == total_files {
                println!("   [{}/{}] files processed...", idx + 1, total_files);
            }
        }
    }

    // Print entity statistics
    println!("\n📊 Code Entity Statistics:");
    let mut type_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (_, entity_type, count) in &all_entities {
        *type_counts.entry(entity_type.as_str()).or_insert(0) += count;
    }
    for (entity_type, count) in &type_counts {
        println!("   {}: {} occurrences", entity_type, count);
    }

    println!("\n🧠 Building knowledge graph...");
    graphrag.build_graph().await?;

    println!("⚡ Loading Candle model (Metal if available)...");
    let mut embedder = CandleEmbedder::new(MODEL_ID)?;
    embedder.initialize().await?;

    let graph = graphrag
        .knowledge_graph_mut()
        .ok_or(GraphRAGError::NotInitialized)?;

    println!("🔢 Embedding {} chunks...", graph.chunks().count());
    let chunk_count = graph.chunks().count();
    for (idx, chunk) in graph.chunks_mut().enumerate() {
        let embedding = embedder.embed(&chunk.content).await?;
        chunk.embedding = Some(embedding);
        if (idx + 1) % 500 == 0 || idx + 1 == chunk_count {
            println!("   [{}/{}] chunks embedded...", idx + 1, chunk_count);
        }
    }

    println!("🔢 Embedding {} entities...", graph.entities().count());
    let entity_count = graph.entities().count();
    for (idx, entity) in graph.entities_mut().enumerate() {
        let text = format!("{} {}", entity.name, entity.entity_type);
        let embedding = embedder.embed(&text).await?;
        entity.embedding = Some(embedding);
        if (idx + 1) % 500 == 0 || idx + 1 == entity_count {
            println!("   [{}/{}] entities embedded...", idx + 1, entity_count);
        }
    }

    let output_path = output_dir.join("graph.json");
    graph.save_to_json(output_path.to_string_lossy().as_ref())?;

    println!(
        "\n✅ Done! docs={}, chunks={}, entities={}, output={}",
        graph.documents().count(),
        graph.chunks().count(),
        graph.entities().count(),
        output_path.display()
    );

    Ok(())
}
