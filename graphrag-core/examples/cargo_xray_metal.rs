///! Analyze a local codebase with GraphRAG + Candle (Metal) embeddings.
///!
///! Usage (from crates/graphrag-rs):
///!   cargo run -p graphrag-core --example cargo_xray_metal --features metal -- ../../..
use graphrag_core::core::error::{GraphRAGError, Result};
use graphrag_core::embeddings::{neural::CandleEmbedder, EmbeddingProvider};
use graphrag_core::{Config, GraphRAG};
use std::collections::VecDeque;
use tracing_subscriber::EnvFilter;
use std::fs;
use std::path::{Path, PathBuf};

const MAX_FILE_BYTES: u64 = 512 * 1024;
const MAX_FILES: usize = 800;
const MODEL_ID: &str = "sentence-transformers/all-MiniLM-L6-v2";

fn should_skip_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | "dist"
            | "build"
            | ".next"
            | ".cache"
            | "vendor"
            | "tmp"
    )
}

fn should_include_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("rs") | Some("toml") | Some("md") | Some("txt") => true,
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

        for entry in entries {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
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
                Ok(metadata) => metadata,
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

fn read_file(path: &Path) -> Option<String> {
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
    // Initialize tracing for device selection visibility
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::new("graphrag_core=info"))
        .init();
    
    let mut args = std::env::args().skip(1);
    let root_arg = args.next().unwrap_or_else(|| ".".to_string());
    let output_arg = args.next().unwrap_or_else(|| "./graphrag-output".to_string());

    let root = PathBuf::from(root_arg);
    let root = root.canonicalize().unwrap_or(root);
    if !root.is_dir() {
        return Err(GraphRAGError::Config {
            message: format!("Root path is not a directory: {}", root.display()),
        });
    }

    let output_dir = PathBuf::from(output_arg);
    fs::create_dir_all(&output_dir)?;

    println!("📂 Scanning {}", root.display());
    let files = collect_files(&root)?;
    println!("✅ Collected {} files (max {})", files.len(), MAX_FILES);

    let mut config = Config::default();
    config.output_dir = output_dir.display().to_string();
    config.embeddings.backend = "candle".to_string();
    config.embeddings.model = Some(MODEL_ID.to_string());
    config.embeddings.fallback_to_hash = false;
    config.approach = "algorithmic".to_string();
    config.entities.use_gleaning = false;
    config.graph.extract_relationships = true;

    let mut graphrag = GraphRAG::new(config)?;
    graphrag.initialize()?;

    for path in &files {
        if let Some(content) = read_file(path) {
            let decorated = format!("// file: {}\n{}", path.display(), content);
            graphrag.add_document_from_text(&decorated)?;
        }
    }

    println!("🧠 Building knowledge graph...");
    graphrag.build_graph().await?;

    println!("⚡ Loading Candle model (Metal if available)...");
    let mut embedder = CandleEmbedder::new(MODEL_ID)?;
    embedder.initialize().await?;

    let graph = graphrag
        .knowledge_graph_mut()
        .ok_or(GraphRAGError::NotInitialized)?;

    for chunk in graph.chunks_mut() {
        let embedding = embedder.embed(&chunk.content).await?;
        chunk.embedding = Some(embedding);
    }

    for entity in graph.entities_mut() {
        let text = format!("{} {}", entity.name, entity.entity_type);
        let embedding = embedder.embed(&text).await?;
        entity.embedding = Some(embedding);
    }

    let output_path = output_dir.join("graph.json");
    graph.save_to_json(output_path.to_string_lossy().as_ref())?;

    println!(
        "📦 Done. docs={}, chunks={}, entities={}, output={}",
        graph.documents().count(),
        graph.chunks().count(),
        graph.entities().count(),
        output_path.display()
    );

    Ok(())
}
