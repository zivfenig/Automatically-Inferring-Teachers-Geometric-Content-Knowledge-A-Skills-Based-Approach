# ============================================================================
# CONFIGURATION FOR RAG-BASED CLASSIFICATION
# ============================================================================
# This file centralizes all paths, credentials, and hyperparameters for the
# classify_van_hiele_gemini_RAG.py script.
# ============================================================================

from pathlib import Path

# ============================================================================
# CORE PATHS
# ============================================================================
# Define BASE as the root of the Retrieval-Augmented-Classification directory
BASE_DIR = Path(__file__).parent

# --- Resource Paths ---
# These files provide the context and knowledge base for the prompts.
ONLY_DEFS_TXT = BASE_DIR / "resources" / "HE_resources" / "only_definitions.txt"
OPERATIVE_DOC = BASE_DIR / "resources" / "HE_resources" / "Operative_doc_short_version.txt"
INDICATORS_PY = BASE_DIR.parent / "Data-and-preprocess" / "HE_Skills_dictionary" / "indicators_dictionary.py"

# --- Data and Embedding Paths ---
# Location of the cross-validation folds and their corresponding embeddings.
DATA_PREPROCESS_DIR = BASE_DIR.parent / "Data-and-preprocess"
FOLDS_DIR = DATA_PREPROCESS_DIR / "HE_Van_Hiele_Dataset" / "folds"
EMB_BASE_DIR = BASE_DIR / "embeddings_folds" / "HE_embedded_folds"

# --- Output Path ---
# Directory where classification results will be saved.
RESULTS_DIR = BASE_DIR / "results" / "gemini"


# ============================================================================
# GCP & VERTEX AI CONFIGURATION
# ============================================================================
# Credentials and model identifiers for Google Cloud Vertex AI.
# IMPORTANT: Replace "xxxxx" with your actual GCP Project ID and Location.
PROJECT_ID = "xxxxx"  # Your GCP project ID
LOCATION = "us-central1"  # Your Vertex AI region (e.g., "us-central1")
MODEL_ID = "publishers/google/models/gemini-2.0-flash"


# ============================================================================
# EXPERIMENT & RETRIEVER HYPERPARAMETERS
# ============================================================================
# --- Experiment Settings ---
# Defines which folds and K-values to run experiments on.
FOLDS_TO_RUN = [1, 2, 3, 4, 5]
K_VALUES_TO_TEST = [5]

# --- Retriever Settings ---
# Alpha controls the weighting between question and answer similarity during retrieval.
# 0.0 = only question, 1.0 = only answer.
ALPHA = 0.8

# The maximum number of examples to fetch from the retriever.
# This should be at least the maximum value in K_VALUES_TO_TEST.
TOP_K_FOR_RETRIEVER = max(K_VALUES_TO_TEST)

# --- Generation Settings ---
# Parameters for the Gemini model's text generation.
MAX_OUTPUT_TOKENS = 300
TEMPERATURE = 0.0
