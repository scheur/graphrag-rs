//! Neural embedding models using Candle for local inference
//!
//! This module provides GPU-accelerated embeddings using Candle with Metal (Apple Silicon)
//! or CUDA (NVIDIA) backends.
//!
//! # Supported Models
//! - all-MiniLM-L6-v2 (384 dimensions, fast)
//! - all-mpnet-base-v2 (768 dimensions, higher quality)
//! - BERT-based sentence transformers
//!
//! # Example
//! ```rust,ignore
//! use graphrag_core::embeddings::neural::CandleEmbedder;
//!
//! let mut embedder = CandleEmbedder::new("sentence-transformers/all-MiniLM-L6-v2")?;
//! embedder.initialize().await?;
//! let embedding = embedder.embed("Hello world").await?;
//! ```

use crate::core::error::GraphRAGError;
use super::EmbeddingProvider;

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::api::tokio::ApiBuilder;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

type Result<T> = std::result::Result<T, GraphRAGError>;
const HF_ENDPOINT: &str = "https://huggingface.co";

/// Convert Candle error to GraphRAGError
fn candle_err(e: candle_core::Error) -> GraphRAGError {
    GraphRAGError::VectorSearch {
        message: format!("Candle error: {}", e),
    }
}

fn hf_hub_err(e: hf_hub::api::tokio::ApiError) -> GraphRAGError {
    GraphRAGError::Embedding {
        message: format!("Hugging Face Hub error: {}", e),
    }
}

fn tokenizer_err(e: tokenizers::Error) -> GraphRAGError {
    GraphRAGError::Embedding {
        message: format!("Tokenizer error: {}", e),
    }
}

#[derive(Debug, Clone, Copy)]
enum WeightsFormat {
    SafeTensors,
    Pytorch,
}

struct ModelFiles {
    weights: PathBuf,
    weights_format: WeightsFormat,
    config: PathBuf,
    tokenizer: PathBuf,
}

/// Static mean pooling - can be used inside spawn_blocking/block_in_place
fn mean_pooling_static(embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    let mask = attention_mask
        .unsqueeze(2).map_err(candle_err)?
        .expand(embeddings.shape()).map_err(candle_err)?
        .to_dtype(embeddings.dtype()).map_err(candle_err)?;
    let masked = embeddings.mul(&mask).map_err(candle_err)?;
    let summed = masked.sum(1).map_err(candle_err)?;
    let mask_sum = mask.sum(1).map_err(candle_err)?
        .clamp(1e-9, f64::MAX).map_err(candle_err)?;
    summed.div(&mask_sum).map_err(candle_err)
}

/// Static L2 normalize - can be used inside spawn_blocking/block_in_place
fn normalize_static(embeddings: &Tensor) -> Result<Tensor> {
    let norm = embeddings.sqr().map_err(candle_err)?
        .sum_keepdim(1).map_err(candle_err)?
        .sqrt().map_err(candle_err)?
        .clamp(1e-12, f64::MAX).map_err(candle_err)?;
    embeddings.broadcast_div(&norm).map_err(candle_err)
}

/// Candle-based neural embedding provider
///
/// Uses Candle for local inference with optional GPU acceleration via Metal or CUDA.
pub struct CandleEmbedder {
    /// Model identifier (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    model_id: String,
    
    /// Embedding dimensions
    dimensions: usize,
    
    /// Cache directory for downloaded models
    cache_dir: PathBuf,
    
    /// Device (CPU, Metal, or CUDA)
    device: Device,

    /// Loaded model (wrapped for async access)
    model: Arc<RwLock<Option<Arc<BertModel>>>>,

    /// Loaded tokenizer (wrapped for async access)
    tokenizer: Arc<RwLock<Option<Arc<Tokenizer>>>>,

    /// Max sequence length for tokenizer truncation
    max_length: usize,

    initialized: bool,
}

impl CandleEmbedder {
    const DEFAULT_MAX_LENGTH: usize = 512;

    /// Create a new Candle embedder
    ///
    /// # Arguments
    /// * `model_id` - Model identifier from Hugging Face (e.g., "sentence-transformers/all-MiniLM-L6-v2")
    ///
    /// # Returns
    /// A new CandleEmbedder instance (not yet initialized)
    pub fn new(model_id: &str) -> Result<Self> {
        let dimensions = Self::guess_dimensions(model_id);
        let cache_dir = Self::default_cache_dir();
        let device = Self::select_device()?;

        Ok(Self {
            model_id: model_id.to_string(),
            dimensions,
            cache_dir,
            device,
            model: Arc::new(RwLock::new(None)),
            tokenizer: Arc::new(RwLock::new(None)),
            max_length: Self::DEFAULT_MAX_LENGTH,
            initialized: false,
        })
    }

    fn guess_dimensions(model_id: &str) -> usize {
        match model_id {
            m if m.contains("MiniLM-L6") => 384,
            m if m.contains("mpnet-base") => 768,
            m if m.contains("all-MiniLM-L12") => 384,
            _ => 384,
        }
    }

    fn default_cache_dir() -> PathBuf {
        if let Ok(dir) = std::env::var("CARGO_XRAY_EMBEDDINGS_DIR") {
            return PathBuf::from(dir);
        }
        if let Ok(dir) = std::env::var("XRAY_EMBEDDINGS_DIR") {
            return PathBuf::from(dir);
        }
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("graphrag")
            .join("models")
    }
    
    /// Select the best available device
    fn select_device() -> Result<Device> {
        // Try Metal first when feature is enabled (Apple Silicon)
        #[cfg(feature = "metal")]
        {
            tracing::info!("Metal feature enabled, attempting GPU initialization...");
            match Device::new_metal(0) {
                Ok(device) => {
                    tracing::info!("Using Metal GPU for embeddings");
                    return Ok(device);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Metal initialization failed, falling back to CPU");
                }
            }
        }
        
        #[cfg(not(feature = "metal"))]
        tracing::warn!("Metal feature not enabled at compile time");
        
        // Try CUDA (NVIDIA)
        #[cfg(feature = "cuda")]
        {
            match Device::new_cuda(0) {
                Ok(device) => {
                    #[cfg(feature = "tracing")]
                    tracing::info!("Using CUDA device for embeddings");
                    return Ok(device);
                }
                Err(_) => {}
            }
        }
        
        // Fall back to CPU
        #[cfg(feature = "tracing")]
        tracing::info!("Using CPU device for embeddings");
        Ok(Device::Cpu)
    }

    /// Try to find model files in local cache directory (hf_hub cache format)
    /// Returns None if not found, allowing fallback to download
    fn find_local_model_files(&self) -> Option<ModelFiles> {
        // hf_hub cache format: models--{org}--{model}/snapshots/{revision}/
        let model_dir_name = format!("models--{}", self.model_id.replace('/', "--"));
        let model_dir = self.cache_dir.join(&model_dir_name);
        
        #[cfg(feature = "tracing")]
        tracing::debug!(model_dir = ?model_dir, "find_local_model_files: checking for local model");
        
        if !model_dir.exists() {
            #[cfg(feature = "tracing")]
            tracing::debug!("find_local_model_files: model directory not found");
            return None;
        }
        
        let snapshots_dir = model_dir.join("snapshots");
        if !snapshots_dir.exists() {
            #[cfg(feature = "tracing")]
            tracing::debug!("find_local_model_files: snapshots directory not found");
            return None;
        }
        
        // Find the first snapshot directory (usually there's only one)
        let snapshot_dir = std::fs::read_dir(&snapshots_dir)
            .ok()?
            .filter_map(|e| e.ok())
            .find(|e| e.path().is_dir())
            .map(|e| e.path())?;
        
        #[cfg(feature = "tracing")]
        tracing::debug!(snapshot_dir = ?snapshot_dir, "find_local_model_files: found snapshot");
        
        let config = snapshot_dir.join("config.json");
        let tokenizer = snapshot_dir.join("tokenizer.json");
        let safetensors = snapshot_dir.join("model.safetensors");
        let pytorch = snapshot_dir.join("pytorch_model.bin");
        
        // Check required files exist
        if !config.exists() || !tokenizer.exists() {
            #[cfg(feature = "tracing")]
            tracing::debug!("find_local_model_files: config.json or tokenizer.json missing");
            return None;
        }
        
        // Prefer safetensors, fall back to pytorch
        let (weights, weights_format) = if safetensors.exists() {
            (safetensors, WeightsFormat::SafeTensors)
        } else if pytorch.exists() {
            (pytorch, WeightsFormat::Pytorch)
        } else {
            #[cfg(feature = "tracing")]
            tracing::debug!("find_local_model_files: no model weights found");
            return None;
        };
        
        #[cfg(feature = "tracing")]
        tracing::info!(
            config = ?config,
            tokenizer = ?tokenizer,
            weights = ?weights,
            "find_local_model_files: found all local model files"
        );
        
        Some(ModelFiles {
            weights,
            weights_format,
            config,
            tokenizer,
        })
    }

    async fn download_model_files(&self) -> Result<ModelFiles> {
        #[cfg(feature = "tracing")]
        tracing::info!(model = %self.model_id, cache_dir = ?self.cache_dir, "download_model_files: resolving model files");
        
        // First, try to find local model files (no network)
        if let Some(files) = self.find_local_model_files() {
            #[cfg(feature = "tracing")]
            tracing::info!("download_model_files: using local model files (no download needed)");
            return Ok(files);
        }
        
        #[cfg(feature = "tracing")]
        tracing::info!("download_model_files: local files not found, downloading from HuggingFace");
        
        let token = std::env::var("HUGGINGFACE_HUB_TOKEN")
            .or_else(|_| std::env::var("HF_TOKEN"))
            .ok();
        
        #[cfg(feature = "tracing")]
        tracing::debug!(has_token = token.is_some(), "download_model_files: HuggingFace auth status");
        
        let api = ApiBuilder::new()
            .with_endpoint(HF_ENDPOINT.to_string())
            .with_cache_dir(self.cache_dir.clone())
            .with_progress(false)
            .with_token(token)
            .build()
            .map_err(hf_hub_err)?;
        let repo = api.model(self.model_id.clone());

        #[cfg(feature = "tracing")]
        tracing::debug!("download_model_files: downloading config.json");
        let config = repo.get("config.json").await.map_err(hf_hub_err)?;
        
        #[cfg(feature = "tracing")]
        tracing::debug!("download_model_files: downloading tokenizer.json");
        let tokenizer = repo
            .get("tokenizer.json")
            .await
            .map_err(hf_hub_err)?;

        #[cfg(feature = "tracing")]
        tracing::debug!("download_model_files: downloading model weights (trying safetensors first)");
        let (weights, weights_format) = match repo.get("model.safetensors").await {
            Ok(path) => {
                #[cfg(feature = "tracing")]
                tracing::debug!(path = ?path, "download_model_files: using safetensors format");
                (path, WeightsFormat::SafeTensors)
            },
            Err(_) => {
                #[cfg(feature = "tracing")]
                tracing::debug!("download_model_files: safetensors not found, trying pytorch_model.bin");
                let path = repo
                    .get("pytorch_model.bin")
                    .await
                    .map_err(hf_hub_err)?;
                #[cfg(feature = "tracing")]
                tracing::debug!(path = ?path, "download_model_files: using pytorch format");
                (path, WeightsFormat::Pytorch)
            }
        };
        
        #[cfg(feature = "tracing")]
        tracing::info!("download_model_files: all model files downloaded");

        Ok(ModelFiles {
            weights,
            weights_format,
            config,
            tokenizer,
        })
    }

    fn load_config(&self, path: &Path) -> Result<BertConfig> {
        let data = std::fs::read_to_string(path).map_err(|e| GraphRAGError::Embedding {
            message: format!("Failed to read model config: {}", e),
        })?;
        serde_json::from_str(&data).map_err(|e| GraphRAGError::Embedding {
            message: format!("Failed to parse model config: {}", e),
        })
    }

    fn load_tokenizer(&self, path: &Path, max_length: usize) -> Result<Tokenizer> {
        let mut tokenizer = Tokenizer::from_file(path).map_err(tokenizer_err)?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(tokenizer_err)?;
        Ok(tokenizer)
    }

    fn load_var_builder(&self, files: &ModelFiles) -> Result<VarBuilder<'static>> {
        match files.weights_format {
            WeightsFormat::SafeTensors => {
                let data = std::fs::read(&files.weights).map_err(|e| GraphRAGError::Embedding {
                    message: format!("Failed to read model weights: {}", e),
                })?;
                VarBuilder::from_buffered_safetensors(data, DTYPE, &self.device)
                    .map_err(candle_err)
            }
            WeightsFormat::Pytorch => VarBuilder::from_pth(&files.weights, DTYPE, &self.device)
                .map_err(candle_err),
        }
    }

    fn load_model(&self, files: &ModelFiles, config: &BertConfig) -> Result<BertModel> {
        let vb = self.load_var_builder(files)?;
        BertModel::load(vb.clone(), config)
            .or_else(|_| BertModel::load(vb.pp("model"), config))
            .map_err(candle_err)
    }
    
    /// Mean pooling over token embeddings
    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // Expand attention mask to match embedding dimensions
        let mask = attention_mask
            .unsqueeze(2).map_err(candle_err)?
            .expand(embeddings.shape()).map_err(candle_err)?
            .to_dtype(embeddings.dtype()).map_err(candle_err)?;
        
        // Apply mask and sum
        let masked = embeddings.mul(&mask).map_err(candle_err)?;
        let summed = masked.sum(1).map_err(candle_err)?;
        
        // Divide by mask sum (avoiding division by zero)
        let mask_sum = mask.sum(1).map_err(candle_err)?
            .clamp(1e-9, f64::MAX).map_err(candle_err)?;
        let pooled = summed.div(&mask_sum).map_err(candle_err)?;
        
        Ok(pooled)
    }
    
    /// L2 normalize embeddings
    fn normalize(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norm = embeddings.sqr().map_err(candle_err)?
            .sum_keepdim(1).map_err(candle_err)?
            .sqrt().map_err(candle_err)?
            .clamp(1e-12, f64::MAX).map_err(candle_err)?;
        let normalized = embeddings.broadcast_div(&norm).map_err(candle_err)?;
        Ok(normalized)
    }

    fn encoding_to_tensors(
        &self,
        encoding: &tokenizers::Encoding,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let input_ids = encoding.get_ids();
        if input_ids.is_empty() {
            return Err(GraphRAGError::VectorSearch {
                message: "Empty text provided".to_string(),
            });
        }
        if input_ids.len() > self.max_length {
            return Err(GraphRAGError::VectorSearch {
                message: format!(
                    "Input exceeds max token length ({})",
                    self.max_length
                ),
            });
        }

        let attention_mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();
        let seq_len = input_ids.len();
        let token_type_ids = if type_ids.len() == seq_len {
            type_ids.to_vec()
        } else {
            vec![0; seq_len]
        };

        let input_ids = Tensor::new(input_ids, &self.device)
            .map_err(candle_err)?
            .reshape((1, seq_len))
            .map_err(candle_err)?;
        let token_type_ids = Tensor::new(token_type_ids.as_slice(), &self.device)
            .map_err(candle_err)?
            .reshape((1, seq_len))
            .map_err(candle_err)?;
        let attention_mask = Tensor::new(attention_mask, &self.device)
            .map_err(candle_err)?
            .reshape((1, seq_len))
            .map_err(candle_err)?;

        Ok((input_ids, token_type_ids, attention_mask))
    }

    /// Generate embedding for text (internal)
    fn embed_internal(
        &self,
        text: &str,
        model: &BertModel,
        tokenizer: &Tokenizer,
    ) -> Result<Vec<f32>> {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(tokenizer_err)?;
        let (input_ids, token_type_ids, attention_mask) =
            self.encoding_to_tensors(&encoding)?;

        let embeddings = model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))
            .map_err(candle_err)?;
        let pooled = self.mean_pooling(&embeddings, &attention_mask)?;
        let normalized = self.normalize(&pooled)?;

        normalized
            .squeeze(0)
            .map_err(candle_err)?
            .to_vec1()
            .map_err(candle_err)
    }
}

#[async_trait::async_trait]
impl EmbeddingProvider for CandleEmbedder {
    async fn initialize(&mut self) -> std::result::Result<(), GraphRAGError> {
        if self.initialized {
            #[cfg(feature = "tracing")]
            tracing::debug!(model = %self.model_id, "initialize: already initialized, skipping");
            return Ok(());
        }

        #[cfg(feature = "tracing")]
        let start_time = std::time::Instant::now();
        
        #[cfg(feature = "tracing")]
        tracing::info!(
            model = %self.model_id,
            cache_dir = ?self.cache_dir,
            device = ?self.device,
            "initialize: starting Candle embedder initialization"
        );

        std::fs::create_dir_all(&self.cache_dir)?;
        
        #[cfg(feature = "tracing")]
        tracing::debug!("initialize: downloading model files");
        let files = self.download_model_files().await?;
        
        #[cfg(feature = "tracing")]
        tracing::debug!("initialize: loading config");
        let config = self.load_config(&files.config)?;
        
        #[cfg(feature = "tracing")]
        tracing::debug!("initialize: loading tokenizer");
        let tokenizer = self.load_tokenizer(&files.tokenizer, config.max_position_embeddings)?;
        
        #[cfg(feature = "tracing")]
        tracing::debug!("initialize: loading model weights to device");
        let model = self.load_model(&files, &config)?;

        self.dimensions = config.hidden_size;
        self.max_length = config.max_position_embeddings;
        *self.model.write().await = Some(Arc::new(model));
        *self.tokenizer.write().await = Some(Arc::new(tokenizer));
        self.initialized = true;

        #[cfg(feature = "tracing")]
        {
            let elapsed = start_time.elapsed();
            tracing::info!(
                model = %self.model_id,
                dimensions = self.dimensions,
                max_length = self.max_length,
                device = ?self.device,
                elapsed_ms = elapsed.as_millis() as u64,
                "initialize: Candle embedder initialized successfully"
            );
        }
        Ok(())
    }
    
    async fn embed(&self, text: &str) -> std::result::Result<Vec<f32>, GraphRAGError> {
        let model = self
            .model
            .read()
            .await
            .as_ref()
            .cloned()
            .ok_or_else(|| GraphRAGError::VectorSearch {
                message: "Model not initialized".to_string(),
            })?;
        let tokenizer = self
            .tokenizer
            .read()
            .await
            .as_ref()
            .cloned()
            .ok_or_else(|| GraphRAGError::VectorSearch {
                message: "Tokenizer not initialized".to_string(),
            })?;

        self.embed_internal(text, model.as_ref(), tokenizer.as_ref())
    }
    
    async fn embed_batch(&self, texts: &[&str]) -> std::result::Result<Vec<Vec<f32>>, GraphRAGError> {
        #[cfg(feature = "tracing")]
        tracing::debug!(
            batch_size = texts.len(),
            model = %self.model_id,
            "embed_batch: starting batch embedding"
        );
        
        if texts.is_empty() {
            #[cfg(feature = "tracing")]
            tracing::debug!("embed_batch: empty input, returning early");
            return Ok(Vec::new());
        }
        
        // For very small batches, use sequential embedding (less overhead)
        if texts.len() <= 2 {
            #[cfg(feature = "tracing")]
            tracing::debug!(batch_size = texts.len(), "embed_batch: using sequential mode for small batch");
            let model = self.model.read().await.as_ref().cloned()
                .ok_or_else(|| GraphRAGError::VectorSearch { message: "Model not initialized".to_string() })?;
            let tokenizer = self.tokenizer.read().await.as_ref().cloned()
                .ok_or_else(|| GraphRAGError::VectorSearch { message: "Tokenizer not initialized".to_string() })?;
            let mut results = Vec::with_capacity(texts.len());
            for text in texts {
                results.push(self.embed_internal(text, model.as_ref(), tokenizer.as_ref())?);
            }
            #[cfg(feature = "tracing")]
            tracing::debug!(count = results.len(), "embed_batch: sequential mode complete");
            return Ok(results);
        }
        
        // True batch embedding for larger batches
        #[cfg(feature = "tracing")]
        tracing::info!(
            batch_size = texts.len(),
            model = %self.model_id,
            "embed_batch: starting GPU/Metal batch processing"
        );
        
        let model = self.model.read().await.as_ref().cloned()
            .ok_or_else(|| GraphRAGError::VectorSearch { message: "Model not initialized".to_string() })?;
        let tokenizer_arc = self.tokenizer.read().await.as_ref().cloned()
            .ok_or_else(|| GraphRAGError::VectorSearch { message: "Tokenizer not initialized".to_string() })?;
        
        // Clone texts for the blocking task
        let texts_owned: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let tokenizer_clone = (*tokenizer_arc).clone();
        let device = self.device.clone();
        let batch_len = texts.len();
        
        #[cfg(feature = "tracing")]
        let start_time = std::time::Instant::now();
        
        // Use block_in_place for synchronous CPU/GPU-bound work to avoid blocking tokio runtime
        // This is critical: model.forward() and tensor ops are sync and can take significant time
        #[cfg(feature = "tracing")]
        tracing::debug!("embed_batch: entering block_in_place for GPU computation");
        
        let result = tokio::task::block_in_place(move || {
            // Clone and configure tokenizer with padding for batch processing
            let mut tokenizer = tokenizer_clone;
            tokenizer.with_padding(Some(PaddingParams {
                strategy: PaddingStrategy::BatchLongest,
                pad_id: 0,
                pad_token: "[PAD]".to_string(),
                ..Default::default()
            }));
            
            // Batch tokenize all texts at once
            #[cfg(feature = "tracing")]
            tracing::debug!(texts_count = texts_owned.len(), "embed_batch: tokenizing batch");
            
            let texts_refs: Vec<&str> = texts_owned.iter().map(|s| s.as_str()).collect();
            let encodings = tokenizer.encode_batch(texts_refs, true).map_err(tokenizer_err)?;
            
            #[cfg(feature = "tracing")]
            tracing::debug!(encodings_count = encodings.len(), "embed_batch: tokenization complete");
            
            let batch_size = encodings.len();
            if batch_size == 0 {
                return Ok(Vec::new());
            }
            
            // Find max sequence length in batch (with padding, all should be same)
            let seq_len = encodings[0].get_ids().len();
            
            // Build batched tensors: (batch_size, seq_len)
            let mut all_input_ids = Vec::with_capacity(batch_size * seq_len);
            let mut all_token_type_ids = Vec::with_capacity(batch_size * seq_len);
            let mut all_attention_mask = Vec::with_capacity(batch_size * seq_len);
            
            for encoding in &encodings {
                let ids = encoding.get_ids();
                let mask = encoding.get_attention_mask();
                let type_ids = encoding.get_type_ids();
                
                if ids.len() != seq_len {
                    return Err(GraphRAGError::VectorSearch {
                        message: format!("Sequence length mismatch: expected {}, got {}", seq_len, ids.len()),
                    });
                }
                
                all_input_ids.extend_from_slice(ids);
                all_attention_mask.extend_from_slice(mask);
                if type_ids.len() == seq_len {
                    all_token_type_ids.extend_from_slice(type_ids);
                } else {
                    all_token_type_ids.extend(std::iter::repeat(0u32).take(seq_len));
                }
            }
            
            // Create tensors with shape (batch_size, seq_len)
            let input_ids = Tensor::new(all_input_ids.as_slice(), &device)
                .map_err(candle_err)?
                .reshape((batch_size, seq_len))
                .map_err(candle_err)?;
            let token_type_ids = Tensor::new(all_token_type_ids.as_slice(), &device)
                .map_err(candle_err)?
                .reshape((batch_size, seq_len))
                .map_err(candle_err)?;
            let attention_mask = Tensor::new(all_attention_mask.as_slice(), &device)
                .map_err(candle_err)?
                .reshape((batch_size, seq_len))
                .map_err(candle_err)?;
            
            // Single forward pass for entire batch!
            #[cfg(feature = "tracing")]
            tracing::debug!(
                batch_size = batch_size,
                seq_len = seq_len,
                "embed_batch: executing model.forward() on GPU/Metal"
            );
            
            let embeddings = model.forward(&input_ids, &token_type_ids, Some(&attention_mask)).map_err(candle_err)?;
            
            #[cfg(feature = "tracing")]
            tracing::debug!("embed_batch: model.forward() complete, pooling embeddings");
            
            // Mean pooling and normalization using static functions
            let pooled = mean_pooling_static(&embeddings, &attention_mask)?;
            let normalized = normalize_static(&pooled)?;
            
            // Extract individual embeddings from batch result
            #[cfg(feature = "tracing")]
            tracing::debug!(batch_size = batch_size, "embed_batch: extracting embeddings from batch result");
            
            let mut results = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let embedding = normalized.get(i).map_err(candle_err)?.to_vec1::<f32>().map_err(candle_err)?;
                results.push(embedding);
            }
            
            #[cfg(feature = "tracing")]
            tracing::debug!(
                embeddings_count = results.len(),
                dimension = results.first().map(|v| v.len()).unwrap_or(0),
                "embed_batch: batch extraction complete"
            );
            
            Ok(results)
        });
        
        #[cfg(feature = "tracing")]
        {
            let elapsed = start_time.elapsed();
            tracing::info!(
                batch_size = batch_len,
                elapsed_ms = elapsed.as_millis() as u64,
                throughput = format!("{:.1} texts/sec", batch_len as f64 / elapsed.as_secs_f64()),
                "embed_batch: GPU/Metal batch complete"
            );
        }
        
        result
    }
    
    fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    fn is_available(&self) -> bool {
        self.initialized
    }
    
    fn provider_name(&self) -> &str {
        "Candle"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_selection() {
        let device = CandleEmbedder::select_device();
        assert!(device.is_ok());
    }
    
    #[test]
    fn test_embedder_creation() {
        let embedder = CandleEmbedder::new("sentence-transformers/all-MiniLM-L6-v2");
        assert!(embedder.is_ok());
        
        let embedder = embedder.unwrap();
        assert_eq!(embedder.dimensions(), 384);
        assert_eq!(embedder.provider_name(), "Candle");
        assert!(!embedder.is_available()); // Not initialized yet
    }
    
    #[test]
    fn test_dimensions_detection() {
        let mini = CandleEmbedder::new("all-MiniLM-L6-v2").unwrap();
        assert_eq!(mini.dimensions(), 384);
        
        let mpnet = CandleEmbedder::new("all-mpnet-base-v2").unwrap();
        assert_eq!(mpnet.dimensions(), 768);
    }
}
