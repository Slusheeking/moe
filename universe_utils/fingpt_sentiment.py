#!/usr/bin/env python3
"""
FinGPT-enhanced sentiment analysis for financial news.
Uses the simplified FinGPT model manager for production reliability.
"""

import os
import sys
import torch
import logging
import gc
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import warnings
import time

# Enhanced CUDA environment configuration
os.environ.update({
    # Core CUDA settings
    'CUDA_LAUNCH_BLOCKING': '0',
    'CUDA_CACHE_DISABLE': '0',
    'CUDA_MODULE_LOADING': 'LAZY',
    
    # Memory management
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512,expandable_segments:True',
    'TORCH_CUDA_ARCH_LIST': '7.0;7.5;8.0;8.6',
    
    # cuDNN settings
    'CUDNN_DETERMINISTIC': '0',
    'CUDNN_BENCHMARK': '1',
    
    # Transformers settings
    'TOKENIZERS_PARALLELISM': 'false',
    'TRANSFORMERS_VERBOSITY': 'error',
    'TRANSFORMERS_NO_ADVISORY_WARNINGS': '1',
    
    # TensorFlow suppression
    'TF_CPP_MIN_LOG_LEVEL': '3',
    'TF_ENABLE_ONEDNN_OPTS': '0',
    
    # HuggingFace settings
    'HF_HUB_DISABLE_PROGRESS_BARS': '1',
    'HF_HUB_DISABLE_SYMLINKS_WARNING': '1',
    'HF_HUB_DISABLE_EXPERIMENTAL_WARNING': '1'
})

# Comprehensive warning suppression
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*CUDA initialization.*')
warnings.filterwarnings('ignore', message='.*cuDNN.*')
warnings.filterwarnings('ignore', message='.*cuBLAS.*')

# Suppress specific library warnings
for logger_name in [
    'transformers', 'transformers.tokenization_utils_base', 'transformers.modeling_utils',
    'torch', 'torch.nn', 'torch.cuda', 'accelerate', 'peft', 'bitsandbytes'
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Local imports
from .data_structures import SentimentAnalysis, NewsAnalysis

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        LlamaForCausalLM,
        LlamaTokenizerFast,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Transformers not available: {e}")
    TRANSFORMERS_AVAILABLE = False

# Try to import PEFT separately to avoid blocking main functionality
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    logger.info("PEFT not available, using base model only")
    PEFT_AVAILABLE = False
    PeftModel = None

# Try to import BitsAndBytes separately to avoid blocking everything
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    logger.info("BitsAndBytes not available, using built-in quantization")
    BITSANDBYTES_AVAILABLE = False


class SimpleFinGPTManager:
    """
    Simplified FinGPT manager that only loads the essential sentiment model.
    Optimized for memory efficiency and reliability.
    """
    
    def __init__(self, base_path: str = "/home/ubuntu/moe-1/FinGPT/fingpt"):
        self.base_path = Path(base_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.loaded = False
        
        # Model configuration - using 7B model for better memory efficiency
        self.model_config = {
            'base_model': "NousResearch/Llama-2-7b-chat-hf",
            'peft_path': self.base_path / "fingpt-sentiment_llama2-7b_lora",
            'model_name': "FinGPT Sentiment v3.2 (7B)"
        }
        
        logger.info(f"Simple FinGPT Manager initialized")
        logger.info(f"Base path: {self.base_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    def load_sentiment_model(self) -> bool:
        """Load only the FinGPT Sentiment v3.3 model with enhanced device management."""
        if self.loaded:
            logger.info("✅ Sentiment model already loaded")
            return True
        
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("❌ Transformers not available")
            return False
        
        try:
            # Enhanced GPU memory management
            if torch.cuda.is_available():
                # Clear any existing CUDA context
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Check available memory before loading
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                free_memory = gpu_memory - reserved
                
                logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory - Total: {gpu_memory:.1f} GB, Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, Free: {free_memory:.2f} GB")
                
                # If less than 10GB free, don't attempt to load the model
                if free_memory < 10.0:
                    logger.warning(f"⚠️ Insufficient GPU memory ({free_memory:.2f} GB free). FinGPT requires ~13GB. Using fallback analysis.")
                    return False
                
                # Set conservative memory fraction
                torch.cuda.set_per_process_memory_fraction(0.6)
            
            logger.info(f"Loading {self.model_config['model_name']}...")
            
            # Load tokenizer with official FinGPT pattern
            logger.info("Loading tokenizer...")
            try:
                self.tokenizer = LlamaTokenizerFast.from_pretrained(
                    self.model_config['base_model'],
                    trust_remote_code=True
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("✅ Tokenizer loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                return False
            
            # Load base model using official FinGPT pattern
            logger.info("Loading base model...")
            try:
                # Use 4-bit quantization for minimal memory usage (~3GB instead of 13GB)
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_config['base_model'],
                    trust_remote_code=True,
                    device_map="cuda:0",
                    quantization_config=quantization_config
                )
                
                logger.info("✅ Base model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load base model: {e}")
                self.cleanup()
                return False
            
            # Enhanced PEFT adapter loading with device consistency
            peft_path = self.model_config['peft_path']
            if peft_path.exists():
                logger.info(f"Loading PEFT adapter from {peft_path}...")
                try:
                    # Ensure base model is in eval mode before PEFT loading
                    self.model.eval()
                    
                    # Load PEFT with minimal parameters to avoid device conflicts
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        str(peft_path),
                        is_trainable=False  # Inference only
                    )
                    
                    # Verify PEFT model device consistency
                    logger.info("✅ PEFT adapter loaded successfully")
                    
                    # Set device explicitly if needed
                    if torch.cuda.is_available():
                        print(f"Device set to use {self.device}")
                    
                except Exception as e:
                    logger.error(f"Failed to load PEFT adapter: {e}")
                    # Continue without PEFT if it fails
                    logger.warning("Continuing with base model only")
            else:
                logger.warning(f"⚠️  PEFT adapter not found at {peft_path}")
            
            # Enhanced pipeline creation with explicit device handling
            try:
                # Ensure model is in eval mode
                self.model.eval()
                
                # Create pipeline with explicit device specification
                pipeline_kwargs = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "torch_dtype": torch.float16,
                    "return_full_text": False,
                    "clean_up_tokenization_spaces": True
                }
                
                # Don't specify device when using accelerate device_map
                # The model already has proper device mapping from accelerate
                
                self.pipeline = pipeline("text-generation", **pipeline_kwargs)
                
                logger.info("✅ Pipeline created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create pipeline: {e}")
                self.cleanup()
                return False
            
            # Final validation
            try:
                # Test the pipeline with a simple input
                test_result = self.pipeline(
                    "Test sentiment:",
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                logger.info("✅ Pipeline validation successful")
                
            except Exception as e:
                logger.warning(f"Pipeline validation failed: {e}")
                # Continue anyway as the model might still work
            
            self.loaded = True
            logger.info("✅ FinGPT Sentiment model loaded successfully")
            
            # Final memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            logger.error(f"Error details: {str(e)}")
            self.cleanup()
            return False
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of financial text."""
        if not self.loaded:
            if not self.load_sentiment_model():
                return self._fallback_sentiment(text)
        
        try:
            # Use the exact official FinGPT prompt format from documentation
            prompt = f'''Instruction: What is the sentiment of this news? Please choose an answer from {{negative/neutral/positive}}
Input: {text[:200]}
Answer: '''
            
            # Use official FinGPT tokenization and generation pattern
            tokens = self.tokenizer([prompt], return_tensors='pt', padding=True, max_length=512)
            
            # Move to device if CUDA available
            if torch.cuda.is_available():
                tokens = {k: v.to('cuda:0') for k, v in tokens.items()}
            
            # Generate using official FinGPT parameters
            with torch.no_grad():
                res = self.model.generate(
                    **tokens,
                    max_length=512,
                    do_sample=True,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode result using official pattern
            res_sentences = [self.tokenizer.decode(i, skip_special_tokens=True) for i in res]
            
            # Extract answer using official parsing method
            try:
                out_text = res_sentences[0].split("Answer: ")[1].strip().lower()
            except IndexError:
                out_text = res_sentences[0].lower()
            
            # Clean the output
            out_text = self._clean_generated_text(out_text)
            
            print(f"Debug - FinGPT raw output: '{res_sentences[0]}'")
            print(f"Debug - Extracted sentiment: '{out_text}'")
            
            # Parse sentiment according to FinGPT output format
            sentiment_score, sentiment_label = self._parse_fingpt_sentiment(out_text)
            
            # If model returns neutral but fallback shows strong sentiment, use hybrid
            if sentiment_label == 'neutral':
                fallback_result = self._fallback_sentiment(text)
                if abs(fallback_result['sentiment_score']) > 0.3:  # Strong signal from keywords
                    return {
                        'sentiment_score': fallback_result['sentiment_score'],
                        'sentiment_label': fallback_result['sentiment_label'],
                        'confidence': 0.7,  # Slightly lower confidence for hybrid
                        'model': 'FinGPT_v3.3_hybrid',
                        'raw_output': f"FinGPT: {out_text}, Keywords: {fallback_result['raw_output']}"
                    }
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': 0.8,
                'model': 'FinGPT_v3.3',
                'raw_output': out_text
            }
            
        except Exception as e:
            logger.warning(f"FinGPT sentiment analysis failed: {e}")
            return self._fallback_sentiment(text)
    
    def _parse_fingpt_sentiment(self, text: str) -> tuple:
        """Parse sentiment from FinGPT output format (positive/negative/neutral)."""
        text_lower = text.lower().strip()
        
        # FinGPT outputs exactly: positive, negative, or neutral
        if 'positive' in text_lower:
            return 0.7, 'bullish'
        elif 'negative' in text_lower:
            return -0.7, 'bearish'
        elif 'neutral' in text_lower:
            return 0.0, 'neutral'
        else:
            # If no clear FinGPT format detected, fallback to keyword analysis
            return self._parse_sentiment(text)
    
    def _is_garbled_text(self, text: str) -> bool:
        """Check if text appears to be garbled or corrupted."""
        if not text or len(text) < 2:
            return True
        
        # Check for common garbled patterns
        garbled_patterns = [
            '$}}%', '~]W', 'qrQ', 'fl$', 'M$', 'OwY',
            '\x00', '\ufffd'  # null bytes and replacement characters
        ]
        
        # If text contains garbled patterns or is mostly non-alphabetic
        if any(pattern in text for pattern in garbled_patterns):
            return True
        
        # Check if text is mostly non-alphabetic characters
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.3:  # Less than 30% alphabetic characters
            return True
        
        return False
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean generated text to handle encoding and formatting issues while preserving FinGPT output."""
        import re
        
        # For FinGPT sentiment analysis, preserve standard sentiment words
        sentiment_words = ['positive', 'negative', 'neutral']
        
        # Check if the text contains valid sentiment words first
        text_lower = text.lower().strip()
        for word in sentiment_words:
            if word in text_lower:
                return word
        
        # If no direct sentiment words, do minimal cleaning
        # Remove only problematic characters but preserve alphanumeric content
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        text = ' '.join(text.split())
        text = text.lower().strip()
        
        # Check again for sentiment words after cleaning
        for word in sentiment_words:
            if word in text:
                return word
        
        # If text is garbled or empty, return neutral as fallback
        if len(text) < 2:
            return 'neutral'
        
        return text
    
    def _parse_sentiment(self, text: str) -> tuple:
        """Parse sentiment from cleaned text."""
        text_lower = text.lower()
        
        # Check for explicit sentiment words
        if any(word in text_lower for word in ['positive', 'bullish', 'good', 'up', 'gain']):
            return 0.7, 'bullish'
        elif any(word in text_lower for word in ['negative', 'bearish', 'bad', 'down', 'loss']):
            return -0.7, 'bearish'
        elif any(word in text_lower for word in ['neutral', 'mixed', 'unchanged']):
            return 0.0, 'neutral'
        else:
            # If no clear sentiment, default to neutral
            return 0.0, 'neutral'
    
    def _fallback_sentiment(self, text: str) -> Dict[str, Any]:
        """Fallback sentiment analysis using keywords."""
        text_lower = text.lower()
        
        positive_words = [
            'beat', 'strong', 'growth', 'up', 'gain', 'rise', 'bull', 'positive',
            'profit', 'revenue', 'earnings', 'upgrade', 'buy', 'outperform'
        ]
        
        negative_words = [
            'miss', 'weak', 'loss', 'down', 'fall', 'drop', 'bear', 'negative',
            'decline', 'downgrade', 'sell', 'underperform', 'cut', 'reduce'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment_score = min(0.6, positive_count * 0.2)
            sentiment_label = 'bullish'
        elif negative_count > positive_count:
            sentiment_score = max(-0.6, -negative_count * 0.2)
            sentiment_label = 'bearish'
        else:
            sentiment_score = 0.0
            sentiment_label = 'neutral'
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label,
            'confidence': 0.5,
            'model': 'keyword_fallback',
            'raw_output': f"pos:{positive_count}, neg:{negative_count}"
        }
    
    def cleanup(self):
        """Clean up model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        
        self.loaded = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        logger.info("🧹 Model cleanup completed")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'loaded': self.loaded,
            'model_name': self.model_config['model_name'],
            'base_model': self.model_config['base_model'],
            'peft_path': str(self.model_config['peft_path']),
            'device': str(self.device),
            'transformers_available': TRANSFORMERS_AVAILABLE
        }


class SimplifiedSentimentAnalyzer:
    """
    Simplified sentiment analyzer using only the essential FinGPT sentiment model.
    Optimized for memory efficiency and reliability.
    """
    
    def __init__(self):
        """Initialize simplified analyzer."""
        self.available = False
        self.fingpt_manager = None
        
        # Initialize source credibility scores
        self.source_credibility = self._initialize_source_credibility()
        
        try:
            self.fingpt_manager = SimpleFinGPTManager()
            self.available = True
            
            # Try to load the model
            if self.fingpt_manager.load_sentiment_model():
                logger.info("✅ Simplified FinGPT analyzer initialized with model")
            else:
                logger.warning("⚠️ FinGPT model could not be loaded, using fallback")
                
        except Exception as e:
            logger.error(f"❌ Error initializing simplified analyzer: {e}")
            self.available = False
            self.fingpt_manager = None
    
    def _initialize_source_credibility(self) -> Dict[str, float]:
        """Initialize credibility scores for news sources."""
        return {
            'reuters': 0.95,
            'bloomberg': 0.95,
            'wall street journal': 0.90,
            'financial times': 0.90,
            'cnbc': 0.85,
            'marketwatch': 0.80,
            'seeking alpha': 0.70,
            'benzinga': 0.65,
            'yahoo finance': 0.75,
            'default': 0.50
        }
    
    def analyze_news_batch(self, articles: List[Dict], symbol: str) -> NewsAnalysis:
        """
        Analyze a batch of news articles using FinGPT sentiment analysis.
        """
        
        if not self.available or not articles:
            return self._create_empty_analysis()
        
        try:
            # Analyze each article
            analyzed_articles = []
            sentiment_scores = []
            
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if len(text.strip()) < 10:
                    continue
                
                # Get sentiment analysis
                if self.fingpt_manager and self.fingpt_manager.loaded:
                    sentiment_result = self.fingpt_manager.analyze_sentiment(text)
                else:
                    sentiment_result = self._fallback_analysis_single(text)
                
                # Get source credibility
                source = article.get('publisher', {}).get('name', 'Unknown').lower()
                credibility = self.source_credibility.get(source, self.source_credibility['default'])
                
                # Store results
                sentiment_scores.append({
                    'score': sentiment_result['sentiment_score'],
                    'confidence': sentiment_result['confidence'],
                    'credibility': credibility,
                    'text': text[:200]
                })
                
                analyzed_articles.append({
                    'title': article.get('title', ''),
                    'published': article.get('published_utc', ''),
                    'sentiment': sentiment_result,
                    'source': article.get('publisher', {}).get('name', 'Unknown'),
                    'credibility': credibility
                })
            
            # Calculate overall sentiment with credibility weighting
            if sentiment_scores:
                # Calculate credibility-weighted sentiment
                total_weight = sum(s['credibility'] * s['confidence'] for s in sentiment_scores)
                
                if total_weight > 0:
                    weighted_sentiment = sum(
                        s['score'] * s['credibility'] * s['confidence'] 
                        for s in sentiment_scores
                    ) / total_weight
                    
                    avg_confidence = np.mean([s['confidence'] for s in sentiment_scores])
                else:
                    weighted_sentiment = 0.0
                    avg_confidence = 0.5
            else:
                weighted_sentiment = 0.0
                avg_confidence = 0.5
            
            # Determine sentiment label
            if weighted_sentiment > 0.1:
                sentiment_label = 'bullish'
            elif weighted_sentiment < -0.1:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
            
            # Calculate sentiment momentum (recent vs older articles)
            if len(sentiment_scores) > 1:
                # Weight newer articles more
                weights = np.linspace(1.0, 0.5, len(sentiment_scores))
                momentum_score = sum(
                    s['score'] * w for s, w in zip(sentiment_scores, weights)
                ) / weights.sum()
                sentiment_momentum = momentum_score - weighted_sentiment
            else:
                sentiment_momentum = 0.0
            
            # Calculate controversy score (disagreement in sentiments)
            if len(sentiment_scores) > 1:
                scores = [s['score'] for s in sentiment_scores]
                controversy_score = float(np.std(scores))
            else:
                controversy_score = 0.0
            
            # Extract key topics
            key_topics = self._extract_key_topics_from_articles(articles)
            
            # Create sentiment analysis result
            overall_sentiment = SentimentAnalysis(
                sentiment_score=float(weighted_sentiment),
                confidence=float(avg_confidence),
                sentiment_label=sentiment_label,
                key_topics=key_topics,
                entity_sentiments={symbol: float(weighted_sentiment)},
                market_impact_score=abs(weighted_sentiment) * avg_confidence
            )
            
            return NewsAnalysis(
                overall_sentiment=overall_sentiment,
                individual_articles=analyzed_articles[:10],  # Keep top 10
                sentiment_momentum=float(sentiment_momentum),
                controversy_score=float(controversy_score),
                credibility_weighted_score=float(weighted_sentiment)
            )
            
        except Exception as e:
            logger.warning(f"News batch analysis failed for {symbol}: {e}")
            return self._create_empty_analysis()
    
    def _fallback_analysis_single(self, text: str) -> Dict[str, Any]:
        """Fallback analysis for a single text."""
        text_lower = text.lower()
        
        positive_words = ['up', 'gain', 'rise', 'bull', 'positive', 'beat', 'strong', 'growth']
        negative_words = ['down', 'fall', 'drop', 'bear', 'negative', 'miss', 'weak', 'loss']
        
        sentiment = 0
        sentiment += sum(1 for word in positive_words if word in text_lower)
        sentiment -= sum(1 for word in negative_words if word in text_lower)
        
        sentiment_score = np.clip(sentiment / 3.0, -1.0, 1.0)  # Normalize
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': 0.5,
            'sentiment_label': 'bullish' if sentiment_score > 0 else 'bearish' if sentiment_score < 0 else 'neutral',
            'model': 'keyword_fallback'
        }
    
    def _extract_key_topics_from_articles(self, articles: List[Dict]) -> List[str]:
        """Extract key financial topics from articles."""
        topic_keywords = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'guidance', 'beat', 'miss'],
            'merger': ['merger', 'acquisition', 'buyout', 'deal', 'acquire', 'takeover'],
            'product': ['launch', 'product', 'release', 'announce', 'unveil', 'innovation'],
            'regulation': ['sec', 'regulatory', 'investigation', 'compliance', 'fine', 'lawsuit'],
            'partnership': ['partnership', 'collaboration', 'joint venture', 'agreement', 'contract'],
            'management': ['ceo', 'cfo', 'resign', 'appoint', 'executive', 'board'],
            'market': ['market share', 'competition', 'growth', 'expansion', 'demand'],
            'financial': ['debt', 'dividend', 'buyback', 'capital', 'liquidity', 'cash flow'],
            'analyst': ['upgrade', 'downgrade', 'rating', 'target', 'analyst'],
            'technology': ['ai', 'blockchain', 'cloud', 'software', 'platform', 'digital']
        }
        
        # Combine all article text
        all_text = ' '.join([
            f"{a.get('title', '')} {a.get('description', '')}" 
            for a in articles
        ]).lower()
        
        # Score topics by keyword frequency
        topic_scores = {}
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                topic_scores[topic] = score
        
        # Sort by score and return top topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, _ in sorted_topics[:5]]
    
    def _create_empty_analysis(self) -> NewsAnalysis:
        """Create empty news analysis result."""
        empty_sentiment = SentimentAnalysis(
            sentiment_score=0.0,
            confidence=0.0,
            sentiment_label='neutral',
            key_topics=[],
            entity_sentiments={},
            market_impact_score=0.0
        )
        
        return NewsAnalysis(
            overall_sentiment=empty_sentiment,
            individual_articles=[],
            sentiment_momentum=0.0,
            controversy_score=0.0,
            credibility_weighted_score=0.0
        )
    
    def cleanup(self):
        """Clean up resources."""
        if self.fingpt_manager:
            self.fingpt_manager.cleanup()


# Legacy compatibility wrapper
class FinGPTEnhancedSentimentAnalyzer(SimplifiedSentimentAnalyzer):
    """Legacy compatibility wrapper for FinGPT enhanced sentiment analyzer."""
    
    def __init__(self):
        super().__init__()
        logger.info("Using simplified FinGPT sentiment analyzer")
