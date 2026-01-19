//! GraphRAG indexer for mighty-wings monorepo
//!
//! Indexes Rust, TypeScript, and Swift code with:
//! - Multi-language entity extraction (FUNCTION, STRUCT, CLASS, PROTOCOL, etc.)
//! - Metal GPU-accelerated embeddings via Candle
//! - Leiden community detection
//! - Optional Qdrant vector storage
//!
//! Usage (from crates/graphrag-rs):
//!   # With Qdrant
//!   cargo run -p graphrag-core --example mighty_wings_graphrag --features "metal,leiden" --release -- \
//!     --root /Users/scheur/dev/Github/mighty-wings \
//!     --output /Users/scheur/dev/Github/mighty-wings/target/graphrag \
//!     --qdrant http://localhost:6334 \
//!     --collection mighty-wings-graphrag
//!
//!   # JSON only (no Qdrant)
//!   cargo run -p graphrag-core --example mighty_wings_graphrag --features "metal,leiden" --release -- \
//!     --root /Users/scheur/dev/Github/mighty-wings \
//!     --output /Users/scheur/dev/Github/mighty-wings/target/graphrag

use futures::StreamExt;
use graphrag_core::core::error::{GraphRAGError, Result};
use graphrag_core::embeddings::{neural::CandleEmbedder, EmbeddingProvider};
use graphrag_core::nlp::custom_ner::{CustomNER, EntityType, ExtractionRule, RuleType};
use graphrag_core::{Config, GraphRAG};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, UpsertPointsBuilder, Value as QdrantValue,
    VectorParamsBuilder,
};
use qdrant_client::Qdrant;
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

// =============================================================================
// Configuration
// =============================================================================

const MAX_FILE_BYTES: u64 = 512 * 1024; // 512KB
const MAX_FILES: usize = 5000;
const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

/// Namespace UUID for GraphRAG point IDs (DNS namespace from RFC 4122)
const GRAPHRAG_NAMESPACE: Uuid = uuid::uuid!("6ba7b810-9dad-11d1-80b4-00c04fd430c8");

/// Generate deterministic UUID for a chunk (idempotent across re-runs)
fn chunk_id(file_path: &str, chunk_index: usize) -> String {
    let input = format!("chunk:{}:{}", file_path, chunk_index);
    Uuid::new_v5(&GRAPHRAG_NAMESPACE, input.as_bytes()).to_string()
}

/// Generate deterministic UUID for an entity (idempotent across re-runs)
fn entity_id(name: &str, entity_type: &str) -> String {
    let input = format!("entity:{}:{}", name, entity_type);
    Uuid::new_v5(&GRAPHRAG_NAMESPACE, input.as_bytes()).to_string()
}

/// Directories to always skip
const SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".cache",
    "vendor",
    "tmp",
    "refs",
    "wings-eyes", // Explicitly excluded per user request
];

/// Include directories from build-parallel.sh + frontends
const INCLUDE_DIRS: &[&str] = &[
    // Build targets from build-parallel.sh
    "apps/wings-api-gateway",
    "crates/wings-shadow",
    "crates/ooxml-rust-sdk",
    "services/wingsai-mcp",
    "crates/mighty-wings-web",
    "crates/fin-engine-wings",
    "crates/economic-stats-extractor",
    "crates/busdev-wings",
    // Frontends
    "apps/wings-frontend",
    "apps/wings-landing",
    "crates/wings-frontend-ai",
    // iOS
    "apps/wingsai-ios",
    // Infrastructure
    "crates/mighty-wings-proto",
    "crates/wings-config",
    "crates/wings-runtime",
    // gRPC services
    "services/graphrag-grpc-service",
];

// =============================================================================
// CLI Arguments
// =============================================================================

#[derive(Debug)]
struct Args {
    root: PathBuf,
    output: PathBuf,
    qdrant_url: Option<String>,
    collection: Option<String>,
    max_files: usize,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1).peekable();
    let mut root = PathBuf::from("/Users/scheur/dev/Github/mighty-wings");
    let mut output = PathBuf::from("/Users/scheur/dev/Github/mighty-wings/target/graphrag");
    let mut qdrant_url = None;
    let mut collection = None;
    let mut max_files = MAX_FILES;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--root" => {
                root = PathBuf::from(args.next().ok_or_else(|| GraphRAGError::Config {
                    message: "--root requires a path".to_string(),
                })?);
            }
            "--output" => {
                output = PathBuf::from(args.next().ok_or_else(|| GraphRAGError::Config {
                    message: "--output requires a path".to_string(),
                })?);
            }
            "--qdrant" => {
                qdrant_url = Some(args.next().ok_or_else(|| GraphRAGError::Config {
                    message: "--qdrant requires a URL".to_string(),
                })?);
            }
            "--collection" => {
                collection = Some(args.next().ok_or_else(|| GraphRAGError::Config {
                    message: "--collection requires a name".to_string(),
                })?);
            }
            "--max-files" => {
                max_files = args
                    .next()
                    .ok_or_else(|| GraphRAGError::Config {
                        message: "--max-files requires a number".to_string(),
                    })?
                    .parse()
                    .map_err(|_| GraphRAGError::Config {
                        message: "--max-files must be a number".to_string(),
                    })?;
            }
            "--help" | "-h" => {
                println!(
                    r#"mighty_wings_graphrag - GraphRAG indexer for mighty-wings monorepo

USAGE:
    mighty_wings_graphrag [OPTIONS]

OPTIONS:
    --root <PATH>         Root directory to scan (default: /Users/scheur/dev/Github/mighty-wings)
    --output <PATH>       Output directory for graph.json (default: <root>/target/graphrag)
    --qdrant <URL>        Qdrant server URL (optional, e.g., http://localhost:6334)
    --collection <NAME>   Qdrant collection name (default: mighty-wings-graphrag)
    --max-files <N>       Maximum files to process (default: 5000)
    --help, -h            Show this help message
"#
                );
                std::process::exit(0);
            }
            _ => {
                // Positional args: root, output
                if root.to_str() == Some("/Users/scheur/dev/Github/mighty-wings") {
                    root = PathBuf::from(&arg);
                } else {
                    output = PathBuf::from(&arg);
                }
            }
        }
    }

    // Default collection name
    if qdrant_url.is_some() && collection.is_none() {
        collection = Some("mighty-wings-graphrag".to_string());
    }

    Ok(Args {
        root,
        output,
        qdrant_url,
        collection,
        max_files,
    })
}

// =============================================================================
// Multi-Language NER
// =============================================================================

/// Build a CustomNER with patterns for Rust, TypeScript, and Swift
fn build_multi_language_ner() -> CustomNER {
    let mut ner = CustomNER::new();

    // Register all entity types
    let entity_types = vec![
        // Rust types
        ("FUNCTION", "Function definitions"),
        ("STRUCT", "Struct definitions"),
        ("ENUM", "Enum definitions"),
        ("TRAIT", "Trait definitions"),
        ("IMPL", "Impl blocks"),
        ("MODULE", "Module definitions"),
        ("CRATE", "Crate references"),
        ("TYPE", "Type alias definitions"),
        ("CONST", "Constant definitions"),
        ("MACRO", "Macro definitions"),
        // TypeScript/JavaScript types
        ("CLASS", "Class definitions"),
        ("INTERFACE", "Interface definitions"),
        ("COMPONENT", "Svelte component exports"),
        // Swift types
        ("PROTOCOL", "Swift protocol definitions"),
        ("EXTENSION", "Swift extension blocks"),
    ];

    for (name, desc) in entity_types {
        ner.register_entity_type(EntityType::new(name.to_string(), desc.to_string()));
    }

    // Rust patterns
    let rust_rules = vec![
        ("rust_function", "FUNCTION", r"(?:pub\s+)?(?:async\s+)?fn\s+([a-z_][a-z0-9_]*)\s*[<(]"),
        ("rust_struct", "STRUCT", r"(?:pub\s+)?struct\s+([A-Z][a-zA-Z0-9]*)"),
        ("rust_enum", "ENUM", r"(?:pub\s+)?enum\s+([A-Z][a-zA-Z0-9]*)"),
        ("rust_trait", "TRAIT", r"(?:pub\s+)?trait\s+([A-Z][a-zA-Z0-9]*)"),
        ("rust_impl", "IMPL", r"impl(?:<[^>]+>)?\s+(?:([A-Z][a-zA-Z0-9]*)\s+for\s+)?([A-Z][a-zA-Z0-9]*)"),
        ("rust_module", "MODULE", r"(?:pub\s+)?mod\s+([a-z_][a-z0-9_]*)"),
        ("rust_crate", "CRATE", r"use\s+([a-z_][a-z0-9_]*)::"),
        ("rust_type", "TYPE", r"(?:pub\s+)?type\s+([A-Z][a-zA-Z0-9]*)\s*(?:<[^>]+>)?\s*="),
        ("rust_const", "CONST", r"(?:pub\s+)?const\s+([A-Z_][A-Z0-9_]*)\s*:"),
        ("rust_macro", "MACRO", r"macro_rules!\s+([a-z_][a-z0-9_]*)"),
    ];

    // TypeScript/Svelte patterns
    let typescript_rules = vec![
        ("ts_function", "FUNCTION", r"(?:export\s+)?(?:async\s+)?function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<(]"),
        ("ts_arrow_fn", "FUNCTION", r"(?:export\s+)?(?:const|let)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>"),
        ("ts_class", "CLASS", r"(?:export\s+)?class\s+([A-Z][a-zA-Z0-9]*)"),
        ("ts_interface", "INTERFACE", r"(?:export\s+)?interface\s+([A-Z][a-zA-Z0-9]*)"),
        ("ts_type", "TYPE", r"(?:export\s+)?type\s+([A-Z][a-zA-Z0-9]*)\s*(?:<[^>]+>)?\s*="),
        ("ts_const", "CONST", r"(?:export\s+)?const\s+([A-Z_][A-Z0-9_]*)\s*(?::\s*[^=]+)?\s*="),
        ("svelte_export", "COMPONENT", r"export\s+let\s+([a-zA-Z_][a-zA-Z0-9_]*)"),
    ];

    // Swift patterns
    let swift_rules = vec![
        ("swift_func", "FUNCTION", r"(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?(?:static\s+)?func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[<(]"),
        ("swift_class", "CLASS", r"(?:public\s+|private\s+|internal\s+|fileprivate\s+|open\s+)?(?:final\s+)?class\s+([A-Z][a-zA-Z0-9]*)"),
        ("swift_struct", "STRUCT", r"(?:public\s+|private\s+|internal\s+)?struct\s+([A-Z][a-zA-Z0-9]*)"),
        ("swift_enum", "ENUM", r"(?:public\s+|private\s+|internal\s+)?enum\s+([A-Z][a-zA-Z0-9]*)"),
        ("swift_protocol", "PROTOCOL", r"(?:public\s+|private\s+|internal\s+)?protocol\s+([A-Z][a-zA-Z0-9]*)"),
        ("swift_extension", "EXTENSION", r"extension\s+([A-Z][a-zA-Z0-9]*)"),
        ("swift_typealias", "TYPE", r"(?:public\s+|private\s+)?typealias\s+([A-Z][a-zA-Z0-9]*)\s*="),
    ];

    // Add all rules
    for (name, entity_type, pattern) in rust_rules
        .into_iter()
        .chain(typescript_rules)
        .chain(swift_rules)
    {
        ner.add_rule(ExtractionRule {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            rule_type: RuleType::Regex,
            pattern: pattern.to_string(),
            min_confidence: if entity_type == "CRATE" { 0.85 } else { 0.95 },
            priority: if entity_type == "IMPL" { 8 } else { 10 },
        });
    }

    ner
}

// =============================================================================
// File Collection
// =============================================================================

fn should_skip_dir(name: &str) -> bool {
    SKIP_DIRS.contains(&name)
}

fn get_file_language(path: &Path) -> Option<&'static str> {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("rs") => Some("rust"),
        Some("toml") => Some("toml"),
        Some("md") => Some("markdown"),
        Some("ts") | Some("tsx") => Some("typescript"),
        Some("js") | Some("jsx") => Some("javascript"),
        Some("svelte") => Some("svelte"),
        Some("swift") => Some("swift"),
        _ => None,
    }
}

fn should_include_file(path: &Path) -> bool {
    get_file_language(path).is_some()
}

fn collect_files(root: &Path, max_files: usize) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let mut queue = VecDeque::new();

    // Start with include directories only
    for include_dir in INCLUDE_DIRS {
        let full_path = root.join(include_dir);
        if full_path.is_dir() {
            queue.push_back(full_path);
        }
    }

    // Also scan root for Cargo.toml, README.md, etc.
    if let Ok(entries) = fs::read_dir(root) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && should_include_file(&path) {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.len() <= MAX_FILE_BYTES {
                        files.push(path);
                    }
                }
            }
        }
    }

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
                // Skip excluded directories
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

            if files.len() >= max_files {
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

// =============================================================================
// Qdrant Integration
// =============================================================================

const VECTOR_DIMENSION: u64 = 384; // all-MiniLM-L6-v2

/// Connect to Qdrant and ensure collection exists
async fn setup_qdrant(url: &str, collection: &str) -> std::result::Result<Qdrant, String> {
    let client = Qdrant::from_url(url)
        .build()
        .map_err(|e| format!("Failed to connect to Qdrant: {}", e))?;

    // Check if collection exists
    let collections = client
        .list_collections()
        .await
        .map_err(|e| format!("Failed to list collections: {}", e))?;

    let exists = collections
        .collections
        .iter()
        .any(|c| c.name == collection);

    if !exists {
        println!("      Creating collection: {}", collection);
        client
            .create_collection(
                CreateCollectionBuilder::new(collection)
                    .vectors_config(VectorParamsBuilder::new(VECTOR_DIMENSION, Distance::Cosine)),
            )
            .await
            .map_err(|e| format!("Failed to create collection: {}", e))?;
    } else {
        println!("      Collection exists: {}", collection);
    }

    Ok(client)
}

/// Upsert chunk embeddings to Qdrant with streaming (memory-efficient for large datasets)
async fn upsert_chunks_streaming(
    client: &Qdrant,
    collection: &str,
    chunks: impl Iterator<Item = (usize, &graphrag_core::core::TextChunk)>,
    total: usize,
) -> std::result::Result<usize, String> {
    let counter = AtomicUsize::new(0);
    let collection = collection.to_string();

    // Collect chunks into tuples with UUIDv5 IDs
    let chunk_tuples: Vec<_> = chunks
        .filter_map(|(idx, chunk)| {
            chunk.embedding.as_ref().map(|emb| {
                let file_path = chunk
                    .content
                    .lines()
                    .next()
                    .and_then(|l| l.strip_prefix("// file: "))
                    .unwrap_or("unknown")
                    .to_string();
                let id = chunk_id(&file_path, idx);
                (id, emb.clone(), file_path, chunk.content.clone(), idx)
            })
        })
        .collect();

    let total_chunks = chunk_tuples.len();

    // Stream in batches of 100 with 4 concurrent uploads
    futures::stream::iter(chunk_tuples)
        .chunks(100)
        .for_each_concurrent(4, |batch| {
            let client = client;
            let collection = &collection;
            let counter = &counter;
            async move {
                let points: Vec<PointStruct> = batch
                    .into_iter()
                    .map(|(id, embedding, file_path, content, chunk_index)| {
                        let mut payload = HashMap::new();
                        payload.insert("file_path".to_string(), QdrantValue::from(file_path));
                        payload.insert(
                            "content".to_string(),
                            QdrantValue::from(content.chars().take(500).collect::<String>()),
                        );
                        payload.insert("chunk_index".to_string(), QdrantValue::from(chunk_index as i64));
                        payload.insert("type".to_string(), QdrantValue::from("chunk"));
                        PointStruct::new(id, embedding, payload)
                    })
                    .collect();

                let batch_size = points.len();
                if let Err(e) = client
                    .upsert_points(UpsertPointsBuilder::new(collection, points))
                    .await
                {
                    eprintln!("      Batch upload failed: {}", e);
                }
                let count = counter.fetch_add(batch_size, Ordering::SeqCst) + batch_size;
                if count % 1000 < 100 || count == total {
                    println!("      [{}/{}] chunks uploaded...", count, total);
                }
            }
        })
        .await;

    Ok(total_chunks)
}

/// Upsert entity embeddings to Qdrant with streaming (memory-efficient for large datasets)
async fn upsert_entities_streaming(
    client: &Qdrant,
    collection: &str,
    entities: impl Iterator<Item = &graphrag_core::core::Entity>,
    total: usize,
) -> std::result::Result<usize, String> {
    let counter = AtomicUsize::new(0);
    let collection = collection.to_string();

    // Collect entities into tuples with UUIDv5 IDs
    let entity_tuples: Vec<_> = entities
        .filter_map(|entity| {
            entity.embedding.as_ref().map(|emb| {
                let id = entity_id(&entity.name, &entity.entity_type);
                let source = entity.id.0.clone();
                (
                    id,
                    emb.clone(),
                    entity.name.clone(),
                    entity.entity_type.clone(),
                    source,
                )
            })
        })
        .collect();

    let total_entities = entity_tuples.len();

    // Stream in batches of 100 with 4 concurrent uploads
    futures::stream::iter(entity_tuples)
        .chunks(100)
        .for_each_concurrent(4, |batch| {
            let client = client;
            let collection = &collection;
            let counter = &counter;
            async move {
                let points: Vec<PointStruct> = batch
                    .into_iter()
                    .map(|(id, embedding, name, entity_type, source_file)| {
                        let mut payload = HashMap::new();
                        payload.insert("name".to_string(), QdrantValue::from(name));
                        payload.insert("entity_type".to_string(), QdrantValue::from(entity_type));
                        payload.insert("source_file".to_string(), QdrantValue::from(source_file));
                        payload.insert("type".to_string(), QdrantValue::from("entity"));
                        PointStruct::new(id, embedding, payload)
                    })
                    .collect();

                let batch_size = points.len();
                if let Err(e) = client
                    .upsert_points(UpsertPointsBuilder::new(collection, points))
                    .await
                {
                    eprintln!("      Batch upload failed: {}", e);
                }
                let count = counter.fetch_add(batch_size, Ordering::SeqCst) + batch_size;
                if count % 1000 < 100 || count == total {
                    println!("      [{}/{}] entities uploaded...", count, total);
                }
            }
        })
        .await;

    Ok(total_entities)
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "graphrag_core=info".to_string()),
        ))
        .init();

    let args = parse_args()?;
    let start_time = Instant::now();

    // Validate root
    let root = args.root.canonicalize().unwrap_or(args.root.clone());
    if !root.is_dir() {
        return Err(GraphRAGError::Config {
            message: format!("Root path is not a directory: {}", root.display()),
        });
    }

    // Create output directory
    fs::create_dir_all(&args.output)?;

    println!("================================================================================");
    println!("GraphRAG Indexer for mighty-wings");
    println!("================================================================================");
    println!("Root:       {}", root.display());
    println!("Output:     {}", args.output.display());
    println!("Max files:  {}", args.max_files);
    if let Some(ref url) = args.qdrant_url {
        println!("Qdrant:     {}", url);
        println!("Collection: {}", args.collection.as_ref().unwrap());
    } else {
        println!("Qdrant:     disabled (JSON only)");
    }
    println!("================================================================================\n");

    // Collect files
    println!("[1/5] Scanning directories...");
    let files = collect_files(&root, args.max_files)?;
    println!("      Found {} files\n", files.len());

    // Count by language
    let mut lang_counts: HashMap<&str, usize> = HashMap::new();
    for file in &files {
        if let Some(lang) = get_file_language(file) {
            *lang_counts.entry(lang).or_insert(0) += 1;
        }
    }
    println!("      Files by language:");
    for (lang, count) in &lang_counts {
        println!("        {}: {}", lang, count);
    }
    println!();

    // Configure GraphRAG
    let mut config = Config::default();
    config.output_dir = args.output.display().to_string();
    config.embeddings.backend = "candle".to_string();
    config.embeddings.model = Some(MODEL_ID.to_string());
    config.embeddings.fallback_to_hash = false;
    config.approach = "algorithmic".to_string();
    config.entities.use_gleaning = false;
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
        "CLASS".to_string(),
        "INTERFACE".to_string(),
        "COMPONENT".to_string(),
        "PROTOCOL".to_string(),
        "EXTENSION".to_string(),
    ];
    config.graph.extract_relationships = true;

    let mut graphrag = GraphRAG::new(config)?;
    graphrag.initialize()?;

    // Build multi-language NER
    let ner = build_multi_language_ner();
    let mut all_entities: Vec<(String, String, String, usize)> = Vec::new(); // (name, type, file, count)

    // Process files
    println!("[2/5] Extracting entities and adding documents...");
    let total_files = files.len();
    for (idx, path) in files.iter().enumerate() {
        if let Some(content) = read_file_content(path) {
            // Extract entities
            let entities = ner.extract(&content);
            let relative_path = path
                .strip_prefix(&root)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            for entity in &entities {
                if let Some(pos) = all_entities
                    .iter()
                    .position(|(n, t, _, _)| n == &entity.text && t == &entity.entity_type)
                {
                    all_entities[pos].3 += 1;
                } else {
                    all_entities.push((
                        entity.text.clone(),
                        entity.entity_type.clone(),
                        relative_path.clone(),
                        1,
                    ));
                }
            }

            // Add document with file path prefix
            let lang = get_file_language(path).unwrap_or("unknown");
            let decorated = format!("// file: {}\n// language: {}\n{}", relative_path, lang, content);
            graphrag.add_document_from_text(&decorated)?;

            if (idx + 1) % 100 == 0 || idx + 1 == total_files {
                println!("      [{}/{}] files processed...", idx + 1, total_files);
            }
        }
    }

    // Print entity statistics
    println!("\n      Entity Statistics:");
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for (_, entity_type, _, count) in &all_entities {
        *type_counts.entry(entity_type.as_str()).or_insert(0) += count;
    }
    let mut sorted_types: Vec<_> = type_counts.iter().collect();
    sorted_types.sort_by(|a, b| b.1.cmp(a.1));
    for (entity_type, count) in sorted_types {
        println!("        {}: {}", entity_type, count);
    }
    println!("      Total unique entities: {}", all_entities.len());
    println!();

    // Build knowledge graph
    println!("[3/5] Building knowledge graph...");
    graphrag.build_graph().await?;

    // Load embedder
    println!("[4/5] Loading Candle model (Metal GPU)...");
    let mut embedder = CandleEmbedder::new(MODEL_ID)?;
    embedder.initialize().await?;

    let graph = graphrag
        .knowledge_graph_mut()
        .ok_or(GraphRAGError::NotInitialized)?;


    // Embed chunks in batches
    const EMBEDDING_BATCH_SIZE: usize = 32;
    let chunk_count = graph.chunks().count();
    println!("      Embedding {} chunks in batches of {}...", chunk_count, EMBEDDING_BATCH_SIZE);
    let chunk_contents: Vec<String> = graph.chunks().map(|c| c.content.clone()).collect();

    for batch_start in (0..chunk_count).step_by(EMBEDDING_BATCH_SIZE) {
        let batch_end = (batch_start + EMBEDDING_BATCH_SIZE).min(chunk_count);
        let batch_texts: Vec<&str> = chunk_contents[batch_start..batch_end].iter().map(|s| s.as_str()).collect();

        let embeddings = embedder.embed_batch(&batch_texts).await?;
        for (i, embedding) in embeddings.into_iter().enumerate() {
            if let Some(chunk) = graph.chunks_mut().nth(batch_start + i) {
                chunk.embedding = Some(embedding);
            }
        }

        if batch_end % 500 == 0 || batch_end == chunk_count {
            println!("      [{}/{}] chunks embedded...", batch_end, chunk_count);
        }
    }

    // Embed entities in batches
    let entity_count = graph.entities().count();
    println!("      Embedding {} entities in batches of {}...", entity_count, EMBEDDING_BATCH_SIZE);
    let entity_texts: Vec<String> = graph.entities().map(|e| format!("{} {}", e.name, e.entity_type)).collect();

    for batch_start in (0..entity_count).step_by(EMBEDDING_BATCH_SIZE) {
        let batch_end = (batch_start + EMBEDDING_BATCH_SIZE).min(entity_count);
        let batch_texts: Vec<&str> = entity_texts[batch_start..batch_end].iter().map(|s| s.as_str()).collect();

        let embeddings = embedder.embed_batch(&batch_texts).await?;
        for (i, embedding) in embeddings.into_iter().enumerate() {
            if let Some(entity) = graph.entities_mut().nth(batch_start + i) {
                entity.embedding = Some(embedding);
            }
        }

        if batch_end % 500 == 0 || batch_end == entity_count {
            println!("      [{}/{}] entities embedded...", batch_end, entity_count);
        }
    }

    // Save to JSON
    println!("\n[5/5] Saving outputs...");
    let graph_path = args.output.join("graph.json");
    graph.save_to_json(graph_path.to_string_lossy().as_ref())?;
    println!("      Saved: {}", graph_path.display());

    // Qdrant storage (if configured)
    let mut qdrant_chunks = 0;
    let mut qdrant_entities = 0;

    if let Some(ref url) = args.qdrant_url {
        let collection = args.collection.as_ref().unwrap();
        println!("\n      Connecting to Qdrant...");

        match setup_qdrant(url, collection).await {
            Ok(client) => {
                // Count chunks with embeddings
                let chunk_count = graph.chunks().filter(|c| c.embedding.is_some()).count();
                println!("      Uploading {} chunks to Qdrant (streaming)...", chunk_count);

                // Stream chunks directly without collecting into Vec
                match upsert_chunks_streaming(
                    &client,
                    collection,
                    graph.chunks().enumerate(),
                    chunk_count,
                )
                .await
                {
                    Ok(count) => {
                        qdrant_chunks = count;
                        println!("      Uploaded {} chunks", count);
                    }
                    Err(e) => println!("      Warning: Failed to upload chunks: {}", e),
                }

                // Count entities with embeddings
                let entity_count = graph.entities().filter(|e| e.embedding.is_some()).count();
                println!("      Uploading {} entities to Qdrant (streaming)...", entity_count);

                // Stream entities directly without collecting into Vec
                match upsert_entities_streaming(&client, collection, graph.entities(), entity_count)
                    .await
                {
                    Ok(count) => {
                        qdrant_entities = count;
                        println!("      Uploaded {} entities", count);
                    }
                    Err(e) => println!("      Warning: Failed to upload entities: {}", e),
                }
            }
            Err(e) => {
                println!("      Warning: Qdrant connection failed: {}", e);
                println!("      Continuing with JSON-only output...");
            }
        }
    }

    // Summary
    let elapsed = start_time.elapsed();
    println!("\n================================================================================");
    println!("Indexing Complete!");
    println!("================================================================================");
    println!("Documents:  {}", graph.documents().count());
    println!("Chunks:     {}", graph.chunks().count());
    println!("Entities:   {}", graph.entities().count());
    if qdrant_chunks > 0 || qdrant_entities > 0 {
        println!("Qdrant:     {} chunks + {} entities", qdrant_chunks, qdrant_entities);
    }
    println!("Time:       {:.1}s", elapsed.as_secs_f64());
    println!("Output:     {}", graph_path.display());
    if let Some(ref url) = args.qdrant_url {
        println!("Qdrant:     {}/collections/{}", url, args.collection.as_ref().unwrap());
    }
    println!("================================================================================");

    Ok(())
}
