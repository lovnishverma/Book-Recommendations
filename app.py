import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import gradio as gr
import pickle
import os
import json
from typing import List, Tuple, Dict, Optional, Set
import re
import logging
from datetime import datetime, timedelta
import warnings
from collections import defaultdict, Counter
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from urllib.parse import quote_plus

warnings.filterwarnings('ignore')

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdvancedGoodreadsRecommender:
    def __init__(self, csv_file: str = "books.csv", cache_dir: str = "cache"):
        """Initialize the advanced Goodreads book recommender system with enhanced features"""
        self.csv_file = csv_file
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Core data structures
        self.df = None
        self.model = None
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.reduced_embeddings = None
        self.book_clusters = None

        # Scalers and processors
        self.rating_scaler = MinMaxScaler()
        self.popularity_scaler = MinMaxScaler()
        self.year_scaler = MinMaxScaler()

        # Caching systems
        self.recommendation_cache = {}
        self.similarity_cache = {}
        self.max_cache_size = 1000
        self.cache_ttl = timedelta(hours=24)

        # Enhanced features
        self.genre_keywords = self._load_genre_keywords()
        self.author_similarity = {}
        self.book_networks = defaultdict(list)

        # Performance tracking
        self.recommendation_stats = defaultdict(int)
        self.search_history = []

        # Load and preprocess data
        self.load_and_preprocess_data()

        # Load or create models
        self.setup_models()

        # Build enhanced features
        self.build_enhanced_features()

    def _load_genre_keywords(self) -> Dict[str, List[str]]:
        """Load genre classification keywords"""
        return {
            'fantasy': ['magic', 'dragon', 'wizard', 'fantasy', 'medieval', 'sword', 'quest', 'adventure'],
            'science_fiction': ['space', 'alien', 'future', 'robot', 'technology', 'dystopian', 'cyberpunk'],
            'romance': ['love', 'romance', 'relationship', 'wedding', 'heart', 'passion', 'dating'],
            'mystery': ['murder', 'detective', 'crime', 'investigation', 'police', 'thriller', 'suspense'],
            'horror': ['horror', 'ghost', 'vampire', 'zombie', 'supernatural', 'dark', 'fear'],
            'historical': ['history', 'war', 'historical', 'century', 'ancient', 'medieval', 'colonial'],
            'biography': ['biography', 'memoir', 'life', 'autobiography', 'personal'],
            'self_help': ['self', 'help', 'motivation', 'success', 'guide', 'improvement', 'habits'],
            'business': ['business', 'management', 'leadership', 'entrepreneur', 'finance', 'marketing'],
            'philosophy': ['philosophy', 'wisdom', 'meaning', 'existence', 'ethics', 'moral'],
            'young_adult': ['teen', 'young', 'high school', 'coming of age', 'adolescent'],
            'childrens': ['children', 'kids', 'child', 'picture book', 'elementary']
        }

    def load_and_preprocess_data(self):
        """Enhanced data loading with better error handling and preprocessing"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df_loaded = False

            for encoding in encodings:
                try:
                    self.df = pd.read_csv(
                        self.csv_file, encoding=encoding, low_memory=False)
                    logger.info(
                        f"Successfully loaded dataset with {encoding} encoding")
                    df_loaded = True
                    break
                except UnicodeDecodeError:
                    continue

            if not df_loaded:
                raise Exception("Failed to load CSV with any encoding")

            logger.info(f"Loaded {len(self.df)} books from Goodreads dataset")

        except FileNotFoundError:
            logger.error(f"Dataset file {self.csv_file} not found!")
            raise FileNotFoundError(
                f"Please ensure {self.csv_file} exists in the current directory")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

        # Enhanced data cleaning
        original_count = len(self.df)

        # Remove books without essential information
        essential_columns = ['title', 'authors']
        for col in essential_columns:
            if col in self.df.columns:
                self.df = self.df.dropna(subset=[col])
                self.df = self.df[self.df[col].astype(str).str.strip() != '']

        # Enhanced numeric field processing
        numeric_fields = {
            'average_rating': (0.0, 5.0),
            'ratings_count': (0, np.inf),
            'work_ratings_count': (0, np.inf),
            'work_text_reviews_count': (0, np.inf),
            'ratings_1': (0, np.inf),
            'ratings_2': (0, np.inf),
            'ratings_3': (0, np.inf),
            'ratings_4': (0, np.inf),
            'ratings_5': (0, np.inf),
            'books_count': (1, np.inf),
            'original_publication_year': (1000, 2024)
        }

        for field, (min_val, max_val) in numeric_fields.items():
            if field in self.df.columns:
                self.df[field] = pd.to_numeric(self.df[field], errors='coerce')
                # Apply reasonable bounds
                if max_val != np.inf:
                    self.df[field] = self.df[field].clip(min_val, max_val)
                self.df[field] = self.df[field].fillna(
                    0 if min_val == 0 else min_val)

        # Enhanced text processing
        text_fields = ['title', 'authors', 'original_title', 'isbn', 'isbn13']
        for field in text_fields:
            if field in self.df.columns:
                self.df[f'{field}_clean'] = self.df[field].fillna(
                    '').apply(self.clean_text)

        # Create comprehensive text for embedding with better weighting
        self.df['combined_text'] = self._create_weighted_text()

        # Enhanced feature engineering
        self.df['popularity_score'] = self.calculate_enhanced_popularity_score()
        self.df['rating_quality_score'] = self.calculate_rating_quality_score()
        self.df['recency_score'] = self.calculate_recency_score()
        self.df['diversity_score'] = self.calculate_diversity_score()

        # Create categorical features
        self.df['publication_year'] = self.df.get(
            'original_publication_year', 0).fillna(0)
        self.df['decade'] = (
            self.df['publication_year'] // 10 * 10).astype(int)
        self.df['century'] = (
            self.df['publication_year'] // 100 * 100).astype(int)

        # Detect potential genres from titles and text
        self.df['detected_genres'] = self.df['combined_text'].apply(
            self.detect_genres)

        # Remove duplicate books (same title + author)
        self.df['title_author_hash'] = (self.df['title_clean'] + '_' + self.df['authors_clean']).apply(
            lambda x: hashlib.md5(x.encode()).hexdigest()
        )
        duplicates = self.df.duplicated(
            subset=['title_author_hash'], keep='first')
        self.df = self.df[~duplicates].copy()

        logger.info(
            f"Processed {len(self.df)} books (removed {original_count - len(self.df)} invalid/duplicate entries)")

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better normalization"""
        if pd.isna(text) or text == '':
            return ""

        text = str(text).lower()

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)

        # Handle special characters more carefully
        # Keep hyphens and apostrophes
        text = re.sub(r'[^\w\s\-\']', ' ', text)

        # Handle contractions
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        # Remove extra spaces and normalize
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _create_weighted_text(self) -> pd.Series:
        """Create weighted text combination for better embeddings"""
        weighted_parts = []

        # Title gets highest weight (3x)
        title_weighted = (self.df['title_clean'] + ' ') * 3
        weighted_parts.append(title_weighted)

        # Authors get medium weight (2x)
        authors_weighted = (self.df['authors_clean'] + ' ') * 2
        weighted_parts.append(authors_weighted)

        # Original title if different (1x)
        if 'original_title_clean' in self.df.columns:
            original_diff = self.df['original_title_clean'] != self.df['title_clean']
            original_weighted = self.df['original_title_clean'].where(
                original_diff, '') + ' '
            weighted_parts.append(original_weighted)

        # Combine all parts
        combined = pd.concat(weighted_parts, axis=1).fillna('').sum(axis=1)
        return combined.str.strip()

    def calculate_enhanced_popularity_score(self) -> np.ndarray:
        """Enhanced popularity calculation with multiple signals"""
        ratings_count = self.df['ratings_count'].fillna(0)
        reviews_count = self.df['work_text_reviews_count'].fillna(0)
        work_ratings_count = self.df.get(
            'work_ratings_count', ratings_count).fillna(0)

        # Logarithmic scaling for different components
        ratings_log = np.log1p(ratings_count)
        reviews_log = np.log1p(reviews_count)
        work_ratings_log = np.log1p(work_ratings_count)

        # Weighted combination
        popularity = (
            0.4 * ratings_log +
            0.3 * work_ratings_log +
            0.2 * reviews_log +
            # Author prolificacy
            0.1 * np.log1p(self.df.get('books_count', 1).fillna(1))
        )

        return popularity

    def calculate_rating_quality_score(self) -> np.ndarray:
        """Calculate rating quality with confidence intervals"""
        rating_cols = ['ratings_5', 'ratings_4',
                       'ratings_3', 'ratings_2', 'ratings_1']
        scores = []

        for idx, row in self.df.iterrows():
            ratings = [row.get(col, 0) for col in rating_cols]
            total_ratings = sum(ratings)

            if total_ratings == 0:
                scores.append(0)
                continue

            # Calculate Wilson score confidence interval for better ranking
            avg_rating = row.get('average_rating', 0)
            n = total_ratings

            if n > 0:
                z = 1.96  # 95% confidence
                p_hat = avg_rating / 5.0  # Normalize to 0-1

                wilson_lower = (
                    p_hat + z*z/(2*n) - z*np.sqrt((p_hat*(1-p_hat) + z*z/(4*n))/n))/(1 + z*z/n)
                confidence_score = wilson_lower * 5.0  # Scale back to 0-5

                # Add bonus for rating distribution quality
                rating_entropy = self._calculate_entropy(ratings)
                distribution_bonus = 1 - rating_entropy / \
                    np.log(5)  # Normalized entropy

                final_score = confidence_score + 0.2 * distribution_bonus
            else:
                final_score = 0

            scores.append(final_score)

        return np.array(scores)

    def _calculate_entropy(self, ratings: List[float]) -> float:
        """Calculate entropy of rating distribution"""
        total = sum(ratings)
        if total == 0:
            return 0

        probabilities = [r/total for r in ratings if r > 0]
        if len(probabilities) <= 1:
            return 0

        entropy = -sum(p * np.log(p) for p in probabilities)
        return entropy

    def calculate_recency_score(self) -> np.ndarray:
        """Calculate recency score favoring newer books"""
        current_year = datetime.now().year
        pub_years = self.df['publication_year'].fillna(1900)

        # Sigmoid function for recency (newer books get higher scores)
        years_ago = current_year - pub_years
        # Sigmoid centered at 10 years ago
        recency_scores = 1 / (1 + np.exp((years_ago - 10) / 10))

        return recency_scores

    def calculate_diversity_score(self) -> np.ndarray:
        """Calculate diversity score based on author's range"""
        author_book_counts = self.df.groupby('authors_clean').size()
        author_diversity = self.df['authors_clean'].map(author_book_counts)

        # More books by author can indicate either prolificacy or oversaturation
        # Use a balanced approach
        diversity_scores = np.log1p(
            author_diversity) / (1 + np.log1p(author_diversity))

        return diversity_scores.values

    def detect_genres(self, text: str) -> List[str]:
        """Detect potential genres from text using keyword matching"""
        detected = []
        text_lower = text.lower()

        for genre, keywords in self.genre_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected.append(genre)

        return detected[:3]  # Limit to top 3 genres

    def setup_models(self):
        """Enhanced model setup with caching and optimization"""
        # Load sentence transformer with error handling
        model_cache_path = self.cache_dir / "sentence_transformer"

        try:
            if model_cache_path.exists():
                self.model = SentenceTransformer(str(model_cache_path))
                logger.info("Loaded cached sentence transformer model")
            else:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model.save(str(model_cache_path))
                logger.info("Downloaded and cached sentence transformer model")
        except Exception as e:
            logger.warning(
                f"Error with sentence transformer: {e}, falling back to TF-IDF only")
            self.model = None

        # Load or create embeddings
        embeddings_file = self.cache_dir / \
            f"{Path(self.csv_file).stem}_embeddings.pkl"
        metadata_file = self.cache_dir / \
            f"{Path(self.csv_file).stem}_metadata.json"

        if self._should_recreate_embeddings(embeddings_file, metadata_file):
            if self.model:
                self.create_embeddings(embeddings_file, metadata_file)
            else:
                logger.warning(
                    "No sentence transformer available, using TF-IDF only")
        else:
            self.load_embeddings(embeddings_file)

        # Setup enhanced TF-IDF
        self.setup_tfidf()

        # Setup dimensionality reduction
        self.setup_dimensionality_reduction()

        # Fit scalers
        self.fit_scalers()

        logger.info("Enhanced models setup completed")

    def _should_recreate_embeddings(self, embeddings_file: Path, metadata_file: Path) -> bool:
        """Check if embeddings need to be recreated"""
        if not embeddings_file.exists() or not metadata_file.exists():
            return True

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Check if dataset has changed
            current_hash = hashlib.md5(str(len(self.df)).encode()).hexdigest()
            if metadata.get('dataset_hash') != current_hash:
                return True

            # Check if embeddings are too old (more than 7 days)
            created_time = datetime.fromisoformat(metadata['created_at'])
            if datetime.now() - created_time > timedelta(days=7):
                return True

            return False

        except Exception as e:
            logger.warning(f"Error checking embedding metadata: {e}")
            return True

    def create_embeddings(self, save_file: Path, metadata_file: Path):
        """Enhanced embedding creation with progress tracking"""
        if not self.model:
            return

        logger.info("Creating enhanced embeddings for all books...")
        book_texts = self.df['combined_text'].tolist()

        # Create embeddings in optimized batches
        batch_size = 500  # Reduced for memory efficiency
        embeddings_list = []

        total_batches = (len(book_texts) + batch_size - 1) // batch_size

        for i in range(0, len(book_texts), batch_size):
            batch = book_texts[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(
                    batch,
                    show_progress_bar=True,
                    batch_size=32,  # Internal batch size
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for better cosine similarity
                )
                embeddings_list.append(batch_embeddings)
                logger.info(
                    f"Processed batch {i//batch_size + 1}/{total_batches}")
            except Exception as e:
                logger.error(
                    f"Error processing batch {i//batch_size + 1}: {e}")
                # Create zero embeddings as fallback
                batch_embeddings = np.zeros((len(batch), 384))
                embeddings_list.append(batch_embeddings)

        self.embeddings = np.vstack(embeddings_list)

        # Save embeddings and metadata
        try:
            with open(save_file, 'wb') as f:
                pickle.dump(self.embeddings, f,
                            protocol=pickle.HIGHEST_PROTOCOL)

            metadata = {
                'created_at': datetime.now().isoformat(),
                'dataset_hash': hashlib.md5(str(len(self.df)).encode()).hexdigest(),
                'embedding_shape': self.embeddings.shape,
                'model_name': 'all-MiniLM-L6-v2'
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Embeddings and metadata saved successfully")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")

    def load_embeddings(self, embeddings_file: Path):
        """Load pre-computed embeddings with validation"""
        try:
            with open(embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

            # Validate embeddings
            if len(self.embeddings) != len(self.df):
                logger.warning("Embeddings size mismatch, recreating...")
                self.create_embeddings(
                    embeddings_file, embeddings_file.with_suffix('.json'))
                return

            logger.info("Successfully loaded pre-computed embeddings")

        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            if self.model:
                self.create_embeddings(
                    embeddings_file, embeddings_file.with_suffix('.json'))

    def setup_tfidf(self):
        """Enhanced TF-IDF setup with optimized parameters"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,  # Increased for better coverage
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams
            min_df=3,            # Slightly higher minimum
            max_df=0.7,          # Lower maximum to exclude very common terms
            sublinear_tf=True,   # Use sublinear tf scaling
            norm='l2'            # L2 normalization
        )

        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.df['combined_text'])
            logger.info(f"TF-IDF matrix created: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Error creating TF-IDF matrix: {e}")
            self.tfidf_matrix = None

    def setup_dimensionality_reduction(self):
        """Setup dimensionality reduction for faster similarity computation"""
        if self.embeddings is not None and len(self.embeddings) > 1000:
            try:
                # Use SVD for dimensionality reduction
                n_components = min(100, self.embeddings.shape[1] // 2)
                self.svd_model = TruncatedSVD(
                    n_components=n_components, random_state=42)
                self.reduced_embeddings = self.svd_model.fit_transform(
                    self.embeddings)

                logger.info(
                    f"Reduced embeddings from {self.embeddings.shape[1]} to {n_components} dimensions")

                # Optional: Create book clusters for recommendations
                if len(self.df) > 100:
                    n_clusters = min(50, len(self.df) // 20)
                    kmeans = KMeans(n_clusters=n_clusters,
                                    random_state=42, n_init=10)
                    self.book_clusters = kmeans.fit_predict(
                        self.reduced_embeddings)
                    logger.info(f"Created {n_clusters} book clusters")

            except Exception as e:
                logger.error(f"Error in dimensionality reduction: {e}")
                self.reduced_embeddings = self.embeddings

    def fit_scalers(self):
        """Fit all scalers with proper error handling"""
        try:
            if len(self.df) > 0:
                # Fit scalers
                self.rating_scaler.fit(self.df[['rating_quality_score']])
                self.popularity_scaler.fit(self.df[['popularity_score']])

                if 'publication_year' in self.df.columns:
                    valid_years = self.df['publication_year'][self.df['publication_year'] > 0]
                    if len(valid_years) > 0:
                        self.year_scaler.fit(valid_years.values.reshape(-1, 1))

                logger.info("All scalers fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting scalers: {e}")

    def build_enhanced_features(self):
        """Build additional features for better recommendations"""
        try:
            # Build author similarity network
            self.build_author_similarity()

            # Build book networks based on similar authors/genres
            self.build_book_networks()

            logger.info("Enhanced features built successfully")
        except Exception as e:
            logger.error(f"Error building enhanced features: {e}")

    def build_author_similarity(self):
        """Build author similarity mappings"""
        author_groups = self.df.groupby('authors_clean')

        for author, group in author_groups:
            if len(group) > 1:  # Authors with multiple books
                similar_books = group.index.tolist()
                self.author_similarity[author] = {
                    'books': similar_books,
                    'avg_rating': group['average_rating'].mean(),
                    'genres': [genre for sublist in group['detected_genres'] for genre in sublist]
                }

    def build_book_networks(self):
        """Build book recommendation networks"""
        for idx, row in self.df.iterrows():
            author = row['authors_clean']
            genres = row['detected_genres']

            # Find books by same author
            if author in self.author_similarity:
                self.book_networks[idx].extend(
                    self.author_similarity[author]['books'][:5])

            # Find books with similar genres
            if genres:
                similar_genre_books = self.df[
                    self.df['detected_genres'].apply(
                        lambda x: bool(set(x) & set(genres)))
                ].index.tolist()[:10]
                self.book_networks[idx].extend(similar_genre_books)

            # Remove duplicates and self
            self.book_networks[idx] = list(
                set(self.book_networks[idx]) - {idx})[:10]

    def enhanced_similarity_search(self, query: str, num_candidates: int = 200) -> Tuple[np.ndarray, Dict]:
        """Enhanced similarity search with multiple algorithms and caching"""
        # Check cache first
        cache_key = hashlib.md5(
            f"{query}_{num_candidates}".encode()).hexdigest()
        if cache_key in self.similarity_cache:
            cached_result, timestamp = self.similarity_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_result

        # Initialize scores
        final_scores = np.zeros(len(self.df))
        score_components = {}

        try:
            # 1. Semantic similarity (if available)
            if self.embeddings is not None:
                query_embedding = self.model.encode(
                    [query], normalize_embeddings=True)

                # Use reduced embeddings for speed if available
                if self.reduced_embeddings is not None:
                    query_reduced = self.svd_model.transform(query_embedding)
                    semantic_scores = cosine_similarity(
                        query_reduced, self.reduced_embeddings)[0]
                else:
                    semantic_scores = cosine_similarity(
                        query_embedding, self.embeddings)[0]

                score_components['semantic'] = semantic_scores
                final_scores += 0.40 * semantic_scores

            # 2. TF-IDF keyword similarity
            if self.tfidf_matrix is not None:
                query_tfidf = self.tfidf_vectorizer.transform([query])
                keyword_scores = cosine_similarity(
                    query_tfidf, self.tfidf_matrix)[0]
                score_components['keyword'] = keyword_scores
                final_scores += 0.25 * keyword_scores

            # 3. Popularity boost
            popularity_scores = self.popularity_scaler.transform(
                self.df[['popularity_score']]
            ).flatten()
            score_components['popularity'] = popularity_scores
            final_scores += 0.15 * popularity_scores

            # 4. Quality boost
            quality_scores = self.rating_scaler.transform(
                self.df[['rating_quality_score']]
            ).flatten()
            score_components['quality'] = quality_scores
            final_scores += 0.12 * quality_scores

            # 5. Recency boost (slight preference for newer books)
            recency_scores = self.df['recency_score'].values
            score_components['recency'] = recency_scores
            final_scores += 0.05 * recency_scores

            # 6. Diversity boost
            diversity_scores = self.df['diversity_score'].values
            score_components['diversity'] = diversity_scores
            final_scores += 0.03 * diversity_scores

            # Cache the result
            result = (final_scores, score_components)
            self.similarity_cache[cache_key] = (result, datetime.now())

            # Manage cache size
            if len(self.similarity_cache) > self.max_cache_size:
                oldest_key = min(self.similarity_cache.keys(),
                                 key=lambda k: self.similarity_cache[k][1])
                del self.similarity_cache[oldest_key]

            return result

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            # Return basic scores as fallback
            return np.random.random(len(self.df)), {}

    def apply_advanced_filters(self,
                               min_rating: float = 0.0,
                               min_ratings_count: int = 0,
                               year_range: Tuple[int, int] = (1000, 2024),
                               author_filter: str = "",
                               genre_filter: List[str] = None,
                               exclude_authors: List[str] = None,
                               language_filter: str = "",
                               max_pages: int = None) -> np.ndarray:
        """Apply comprehensive filtering with advanced options"""
        mask = np.ones(len(self.df), dtype=bool)

        try:
            # Basic filters
            if min_rating > 0:
                mask &= (self.df['average_rating'] >= min_rating)

            if min_ratings_count > 0:
                mask &= (self.df['ratings_count'] >= min_ratings_count)

            # Year range filter
            if year_range != (1000, 2024):
                year_mask = (
                    (self.df['publication_year'] >= year_range[0]) &
                    (self.df['publication_year'] <= year_range[1])
                )
                mask &= year_mask.fillna(False)

            # Author filter (include)
            if author_filter.strip():
                author_mask = self.df['authors'].str.contains(
                    author_filter.strip(), case=False, na=False, regex=False
                )
                mask &= author_mask

            # Author filter (exclude)
            if exclude_authors:
                for exclude_author in exclude_authors:
                    if exclude_author.strip():
                        exclude_mask = ~self.df['authors'].str.contains(
                            exclude_author.strip(), case=False, na=False, regex=False
                        )
                        mask &= exclude_mask

            # Genre filter
            if genre_filter:
                genre_mask = self.df['detected_genres'].apply(
                    lambda genres: any(genre.lower() in [
                                       g.lower() for g in genre_filter] for genre in genres)
                )
                mask &= genre_mask

            # Language filter (if available)
            if language_filter and 'language_code' in self.df.columns:
                language_mask = self.df['language_code'].str.contains(
                    language_filter, case=False, na=False
                )
                mask &= language_mask

            # Page count filter (if available)
            if max_pages and 'num_pages' in self.df.columns:
                page_mask = (
                    (self.df['num_pages'] <= max_pages) |
                    (self.df['num_pages'].isna())
                )
                mask &= page_mask

            return mask

        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            return np.ones(len(self.df), dtype=bool)


def get_enhanced_book_details(self, idx: int) -> Dict:
    """Get comprehensive book information with enhanced metadata"""
    try:
        book = self.df.iloc[idx]

        # Calculate rating distribution
        rating_cols = ['ratings_5', 'ratings_4',
                       'ratings_3', 'ratings_2', 'ratings_1']
        total_ratings = sum([book.get(col, 0) for col in rating_cols])

        rating_breakdown = ""
        rating_percentages = {}

        if total_ratings > 0:
            for star in [5, 4, 3, 2, 1]:
                count = book.get(f'ratings_{star}', 0)
                percentage = (count / total_ratings) * 100
                rating_percentages[f'{star}_star'] = percentage
                rating_breakdown += f"{star}⭐: {percentage:.1f}% ({int(count):,}) "

        # Get similar books from network
        similar_books = []
        if idx in self.book_networks:
            similar_indices = self.book_networks[idx][:3]
            for sim_idx in similar_indices:
                if sim_idx < len(self.df):
                    sim_book = self.df.iloc[sim_idx]
                    similar_books.append({
                        'title': sim_book['title'],
                        'authors': sim_book['authors'],
                        'rating': sim_book.get('average_rating', 0)
                    })

        # --- NEW: Generate purchase links ---
        title = book['title'].strip()
        author = book['authors'].strip()

        # URL-safe query string using quote_plus for proper encoding
        from urllib.parse import quote_plus
        query = quote_plus(f"{title} {author}")

        amazon_url = f"https://www.amazon.com/s?k={query}"
        ebook_url = f"https://annas-archive.org/search?q={query}"
        # --- END NEW ---

        # Enhanced book details
        return {
            'index': idx,
            'book_id': book.get('book_id', 'N/A'),
            'title': book['title'],
            'original_title': book.get('original_title', book['title']),
            'authors': book['authors'],
            'publication_year': int(book.get('publication_year', 0)) if book.get('publication_year', 0) > 0 else 'Unknown',
            'decade': book.get('decade', 'Unknown'),
            'average_rating': float(book.get('average_rating', 0)),
            'ratings_count': int(book.get('ratings_count', 0)),
            'work_text_reviews_count': int(book.get('work_text_reviews_count', 0)),
            'work_ratings_count': int(book.get('work_ratings_count', 0)),
            'isbn': book.get('isbn', 'N/A'),
            'isbn13': book.get('isbn13', 'N/A'),
            'language_code': book.get('language_code', 'Unknown'),
            'num_pages': int(book.get('num_pages', 0)) if pd.notna(book.get('num_pages')) else 'Unknown',
            'image_url': book.get('image_url', ''),
            'small_image_url': book.get('small_image_url', ''),
            'rating_breakdown': rating_breakdown.strip(),
            'rating_percentages': rating_percentages,
            'popularity_score': float(book.get('popularity_score', 0)),
            'rating_quality_score': float(book.get('rating_quality_score', 0)),
            'recency_score': float(book.get('recency_score', 0)),
            'detected_genres': book.get('detected_genres', []),
            'similar_books': similar_books,
            'books_count': int(book.get('books_count', 1)),
            # --- ADDED: Purchase links ---
            'purchase_links': {
                'amazon': amazon_url,
                'ebook_search': ebook_url
            }
        }

    except Exception as e:
        logger.error(f"Error getting book details for index {idx}: {e}")
        return {'error': f'Could not retrieve details for book at index {idx}'}

    def explain_recommendation_enhanced(self, query: str, book_idx: int,
                                        similarity_score: float, score_components: Dict) -> str:
        """Generate detailed explanation for recommendations"""
        try:
            book = self.df.iloc[book_idx]
            explanations = []

            query_lower = query.lower()
            title_lower = book['title'].lower()
            author_lower = book['authors'].lower()

            # Exact matches (highest priority)
            if query_lower == title_lower:
                explanations.append("🎯 Exact title match")
            elif query_lower in title_lower:
                explanations.append("📖 Title contains search term")
            elif query_lower in author_lower:
                explanations.append("👤 Author match")

            # Semantic similarity
            if 'semantic' in score_components:
                sem_score = score_components['semantic'][book_idx]
                if sem_score > 0.8:
                    explanations.append("🧠 Excellent semantic match")
                elif sem_score > 0.6:
                    explanations.append("📚 Strong content similarity")
                elif sem_score > 0.4:
                    explanations.append("🔍 Good content relevance")

            # Keyword matching
            if 'keyword' in score_components:
                kw_score = score_components['keyword'][book_idx]
                if kw_score > 0.3:
                    explanations.append("🔤 Strong keyword match")
                elif kw_score > 0.1:
                    explanations.append("📝 Keyword similarity")

            # Quality indicators
            if book['average_rating'] > 4.4:
                explanations.append("⭐ Outstanding rating (4.4+)")
            elif book['average_rating'] > 4.0:
                explanations.append("⭐ Highly rated (4.0+)")

            # Popularity indicators
            if book['ratings_count'] > 100000:
                explanations.append("🔥 Extremely popular (100K+ ratings)")
            elif book['ratings_count'] > 10000:
                explanations.append("📈 Very popular (10K+ ratings)")
            elif book['ratings_count'] > 1000:
                explanations.append("👥 Popular choice (1K+ ratings)")

            # Genre matching
            detected_genres = book.get('detected_genres', [])
            if detected_genres:
                for genre in detected_genres[:2]:  # Top 2 genres
                    if any(keyword in query_lower for keyword in self.genre_keywords.get(genre, [])):
                        explanations.append(
                            f"🎭 {genre.replace('_', ' ').title()} genre match")
                        break

            # Recency factor
            pub_year = book.get('publication_year', 0)
            if pub_year > 2010:
                explanations.append("🆕 Recent publication")
            elif pub_year > 1990:
                explanations.append("📅 Modern classic")

            # Author prolificacy
            books_count = book.get('books_count', 1)
            if books_count > 20:
                explanations.append("📚 Prolific author")

            # Fallback explanation
            if not explanations:
                explanations.append("🤖 AI content analysis match")

            # Return top explanations with overall score
            top_explanations = explanations[:4]
            score_text = f"(Score: {similarity_score:.3f})"

            return f"{' • '.join(top_explanations)} {score_text}"

        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return f"Content similarity (Score: {similarity_score:.3f})"

    def get_recommendation_statistics(self) -> Dict:
        """Get detailed statistics about the recommendation system"""
        return {
            'total_books': len(self.df),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'cache_size': len(self.recommendation_cache),
            'similarity_cache_size': len(self.similarity_cache),
            'total_searches': sum(self.recommendation_stats.values()),
            'recent_searches': len(self.search_history),
            'author_networks': len(self.author_similarity),
            'book_clusters': len(set(self.book_clusters)) if self.book_clusters is not None else 0
        }

    def recommend_books_enhanced(self,
                                 user_input: str,
                                 num_recommendations: int = 10,
                                 min_rating: float = 0.0,
                                 min_popularity: int = 0,
                                 year_range: Tuple[int, int] = (1000, 2024),
                                 author_filter: str = "",
                                 genre_filter: List[str] = None,
                                 exclude_authors: List[str] = None,
                                 include_explanations: bool = True,
                                 diversity_boost: bool = True,
                                 rerank_by_popularity: bool = False) -> List[Dict]:
        """Enhanced recommendation function with advanced features"""

        if not user_input.strip():
            return [{"error": "Please enter a search query (book title, author, or description)."}]

        # Track search
        self.search_history.append({
            'query': user_input,
            'timestamp': datetime.now(),
            'filters': {
                'min_rating': min_rating,
                'min_popularity': min_popularity,
                'year_range': year_range,
                'author_filter': author_filter,
                'genre_filter': genre_filter
            }
        })
        self.recommendation_stats['total_searches'] += 1

        # Generate comprehensive cache key
        cache_components = [
            user_input, num_recommendations, min_rating, min_popularity,
            str(year_range), author_filter, str(genre_filter),
            str(exclude_authors), diversity_boost, rerank_by_popularity
        ]
        cache_key = hashlib.md5(
            '_'.join(map(str, cache_components)).encode()).hexdigest()

        # Check cache
        if cache_key in self.recommendation_cache:
            cached_result, timestamp = self.recommendation_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                self.recommendation_stats['cache_hits'] += 1
                return cached_result

        try:
            # Apply filters
            filter_mask = self.apply_advanced_filters(
                min_rating=min_rating,
                min_ratings_count=min_popularity,
                year_range=year_range,
                author_filter=author_filter,
                genre_filter=genre_filter,
                exclude_authors=exclude_authors
            )

            if not filter_mask.any():
                error_msg = "No books match your filters. Try relaxing the criteria."
                return [{"error": error_msg}]

            # Get enhanced similarity scores
            similarity_scores, score_components = self.enhanced_similarity_search(
                user_input, num_candidates=min(1000, len(self.df))
            )

            # Apply filters to scores
            filtered_scores = np.where(filter_mask, similarity_scores, -1)

            # Get initial top candidates (more than needed for diversity)
            candidate_multiplier = 3 if diversity_boost else 2
            top_indices = filtered_scores.argsort(
            )[-(num_recommendations * candidate_multiplier):][::-1]
            top_indices = top_indices[filtered_scores[top_indices] > -1]

            if len(top_indices) == 0:
                error_msg = "No similar books found. Try a different search term or adjust filters."
                return [{"error": error_msg}]

            # Apply diversity boosting
            if diversity_boost and len(top_indices) > num_recommendations:
                top_indices = self._apply_diversity_boosting(
                    top_indices, num_recommendations)
            else:
                top_indices = top_indices[:num_recommendations]

            # Optional reranking by popularity
            if rerank_by_popularity:
                popularity_scores = self.df.iloc[top_indices]['popularity_score'].values
                rerank_indices = top_indices[np.argsort(
                    popularity_scores)[::-1]]
                top_indices = rerank_indices

            # Prepare enhanced recommendations
            recommendations = []
            for rank, idx in enumerate(top_indices):
                book_details = self.get_enhanced_book_details(idx)

                if 'error' not in book_details:
                    book_details['rank'] = rank + 1
                    book_details['similarity_score'] = f"{similarity_scores[idx]:.4f}"
                    book_details['raw_score'] = float(similarity_scores[idx])

                    if include_explanations:
                        book_details['explanation'] = self.explain_recommendation_enhanced(
                            user_input, idx, similarity_scores[idx], score_components
                        )

                    # Add score breakdown for transparency
                    book_details['score_breakdown'] = {
                        component: float(scores[idx])
                        for component, scores in score_components.items()
                    }

                recommendations.append(book_details)

            # Cache the results
            self.recommendation_cache[cache_key] = (
                recommendations, datetime.now())

            # Manage cache size
            if len(self.recommendation_cache) > self.max_cache_size:
                oldest_key = min(self.recommendation_cache.keys(),
                                 key=lambda k: self.recommendation_cache[k][1])
                del self.recommendation_cache[oldest_key]

            self.recommendation_stats['successful_searches'] += 1
            return recommendations

        except Exception as e:
            logger.error(f"Error in enhanced recommendation: {str(e)}")
            self.recommendation_stats['failed_searches'] += 1
            return [{"error": f"An error occurred: {str(e)}"}]

    def _apply_diversity_boosting(self, candidate_indices: np.ndarray,
                                  target_count: int) -> np.ndarray:
        """Apply diversity boosting to avoid too many similar books"""
        try:
            selected_indices = []
            candidate_books = self.df.iloc[candidate_indices]

            # Track diversity metrics
            selected_authors = set()
            selected_decades = set()
            selected_genres = set()

            # Always include the top match
            if len(candidate_indices) > 0:
                top_idx = candidate_indices[0]
                selected_indices.append(top_idx)

                top_book = self.df.iloc[top_idx]
                selected_authors.add(top_book['authors_clean'])
                selected_decades.add(top_book.get('decade', 0))
                selected_genres.update(top_book.get('detected_genres', []))

            # Select remaining books with diversity considerations
            for idx in candidate_indices[1:]:
                if len(selected_indices) >= target_count:
                    break

                book = self.df.iloc[idx]
                author = book['authors_clean']
                decade = book.get('decade', 0)
                genres = book.get('detected_genres', [])

                # Calculate diversity score
                diversity_score = 0

                # Author diversity (highest weight)
                if author not in selected_authors:
                    diversity_score += 3

                # Decade diversity
                if decade not in selected_decades:
                    diversity_score += 2

                # Genre diversity
                if not any(genre in selected_genres for genre in genres):
                    diversity_score += 1

                # Accept book based on diversity score and position
                position_bonus = max(
                    0, (len(candidate_indices) - len(selected_indices)) / len(candidate_indices))
                final_score = diversity_score + position_bonus

                # Lower threshold for later candidates to ensure we fill the list
                threshold = max(1, 3 - len(selected_indices) * 0.2)

                if final_score >= threshold or len(selected_indices) < target_count // 2:
                    selected_indices.append(idx)
                    selected_authors.add(author)
                    selected_decades.add(decade)
                    selected_genres.update(genres)

            # If we don't have enough, fill with remaining candidates
            while len(selected_indices) < target_count and len(selected_indices) < len(candidate_indices):
                for idx in candidate_indices:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        break

            return np.array(selected_indices[:target_count])

        except Exception as e:
            logger.error(f"Error in diversity boosting: {e}")
            return candidate_indices[:target_count]

# Initialize the enhanced recommender


def initialize_recommender(csv_file: str = "books.csv") -> AdvancedGoodreadsRecommender:
    """Initialize the recommender with proper error handling"""
    try:
        return AdvancedGoodreadsRecommender(csv_file)
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {e}")
        return None


recommender = initialize_recommender()


def format_enhanced_recommendations(recommendations: List[Dict],
                                    show_debug_info: bool = False) -> str:
    """Format recommendations with enhanced display"""
    if not recommendations:
        return "No recommendations found."

    if "error" in recommendations[0]:
        return f"❌ **Error:** {recommendations[0]['error']}"

    formatted_parts = []

    # Header with summary
    total_books = len(recommendations)
    avg_rating = sum(book.get('average_rating', 0)
                     for book in recommendations) / total_books
    total_ratings = sum(book.get('ratings_count', 0)
                        for book in recommendations)

    header = f"""
## 📚 **Found {total_books} Excellent Book Recommendations**
*Average rating: {avg_rating:.2f}⭐ • Total community ratings: {total_ratings:,}*

---
"""
    formatted_parts.append(header)


# Format each book
    for book in recommendations:
        if 'error' in book:
            continue

    # Main book information
    book_section = f"""
### **{book.get('rank', '?')}. {book['title']}**
{f"*({book['original_title']})*" if book.get('original_title') != book['title'] else ""}

**👤 Author:** {book['authors']}  
**📅 Published:** {book['publication_year']} • **📚 Series books:** {book.get('books_count', 1)}  
**⭐ Rating:** {book['average_rating']:.2f}/5.0 **({book['ratings_count']:,} ratings, {book['work_text_reviews_count']:,} reviews)**  

**🎯 Match Score:** {book['similarity_score']} • **🏆 Quality Score:** {book.get('rating_quality_score', 0):.2f}  
**💡 Why recommended:** {book.get('explanation', 'Content similarity')}

"""

    # Rating distribution
    if book.get('rating_breakdown'):
        book_section += f"**📊 Rating Distribution:** {book['rating_breakdown']}\n\n"

    # Genres
    if book.get('detected_genres'):
        genres_text = ' • '.join(
            [g.replace('_', ' ').title() for g in book['detected_genres'][:3]])
        book_section += f"**🎭 Genres:** {genres_text}\n\n"

    # Additional metadata
    metadata_parts = []
    if book.get('isbn13') != 'N/A':
        metadata_parts.append(f"ISBN13: {book['isbn13']}")
    if book.get('num_pages') != 'Unknown':
        metadata_parts.append(f"{book['num_pages']} pages")
    if book.get('language_code') != 'Unknown':
        metadata_parts.append(f"Language: {book['language_code']}")

    if metadata_parts:
        book_section += f"**📖 Details:** {' • '.join(metadata_parts)}\n\n"

    # 🛒 Get This Book section
    if book.get('purchase_links'):
        amazon_link = book['purchase_links']['amazon']
        ebook_link = book['purchase_links']['ebook_search']
        book_section += f"""**🛒 Get This Book:**  
- [📖 Buy on Amazon]({amazon_link}) — Purchase physical/digital copy  
- [📚 Free eBook Search]({ebook_link}) — Find free digital version

"""

    # Similar books
    if book.get('similar_books'):
        similar_list = []
        for sim_book in book['similar_books'][:2]:  # Show top 2
            similar_list.append(
                f"{sim_book['title']} ({sim_book['rating']:.1f}⭐)")
        if similar_list:
            book_section += f"**🔗 Similar books:** {' • '.join(similar_list)}\n\n"

    # Debug information (optional)
    if show_debug_info and book.get('score_breakdown'):
        breakdown = book['score_breakdown']
        debug_info = []
        for component, score in breakdown.items():
            debug_info.append(f"{component}: {score:.3f}")
        book_section += f"**🔍 Score breakdown:** {' • '.join(debug_info)}\n\n"

    book_section += "---\n"
    formatted_parts.append(book_section)


def get_enhanced_dataset_stats():
    """Get comprehensive enhanced dataset statistics"""
    if recommender is None or recommender.df is None:
        return "❌ Dataset not loaded"

    df = recommender.df
    stats = recommender.get_recommendation_statistics()

    # Enhanced statistics
    total_books = len(df)
    total_authors = df['authors'].nunique()
    avg_rating = df['average_rating'].mean()
    total_ratings = df['ratings_count'].sum()

    # Year statistics
    year_min = df['publication_year'][df['publication_year'] > 0].min()
    year_max = df['publication_year'][df['publication_year'] > 0].max()

    # Genre distribution
    all_genres = []
    for genres in df['detected_genres']:
        all_genres.extend(genres)
    genre_counts = Counter(all_genres)
    top_genres = genre_counts.most_common(5)

    # Rating distribution
    rating_dist = {
        'excellent': len(df[df['average_rating'] >= 4.5]),
        'very_good': len(df[(df['average_rating'] >= 4.0) & (df['average_rating'] < 4.5)]),
        'good': len(df[(df['average_rating'] >= 3.5) & (df['average_rating'] < 4.0)]),
        'average': len(df[(df['average_rating'] >= 3.0) & (df['average_rating'] < 3.5)]),
        'below_average': len(df[df['average_rating'] < 3.0])
    }

    # Popularity tiers
    popularity_tiers = {
        'viral': len(df[df['ratings_count'] >= 100000]),
        'very_popular': len(df[(df['ratings_count'] >= 10000) & (df['ratings_count'] < 100000)]),
        'popular': len(df[(df['ratings_count'] >= 1000) & (df['ratings_count'] < 10000)]),
        'niche': len(df[df['ratings_count'] < 1000])
    }

    return f"""
## 📊 **Enhanced Goodreads Dataset Analytics**

### **📚 Collection Overview**
- **Total Books:** {total_books:,}
- **Unique Authors:** {total_authors:,}
- **Publication Range:** {int(year_min)} - {int(year_max)} ({int(year_max - year_min)} years)
- **Total Community Ratings:** {int(total_ratings):,}
- **Average Rating:** {avg_rating:.2f}/5.0

### **🎭 Top Genres**
{chr(10).join([f"• **{genre.replace('_', ' ').title()}:** {count:,} books" for genre, count in top_genres])}

### **⭐ Rating Quality Distribution**
- **Excellent (4.5+):** {rating_dist['excellent']:,} books ({rating_dist['excellent']/total_books*100:.1f}%)
- **Very Good (4.0-4.5):** {rating_dist['very_good']:,} books ({rating_dist['very_good']/total_books*100:.1f}%)
- **Good (3.5-4.0):** {rating_dist['good']:,} books ({rating_dist['good']/total_books*100:.1f}%)
- **Average (3.0-3.5):** {rating_dist['average']:,} books ({rating_dist['average']/total_books*100:.1f}%)
- **Below Average (<3.0):** {rating_dist['below_average']:,} books ({rating_dist['below_average']/total_books*100:.1f}%)

### **🔥 Popularity Distribution**
- **Viral (100K+ ratings):** {popularity_tiers['viral']:,} books
- **Very Popular (10K-100K):** {popularity_tiers['very_popular']:,} books
- **Popular (1K-10K):** {popularity_tiers['popular']:,} books
- **Niche (<1K):** {popularity_tiers['niche']:,} books

### **🤖 AI System Stats**
- **Embedding Dimensions:** {stats['embedding_dimension']}
- **Book Clusters:** {stats['book_clusters']}
- **Author Networks:** {stats['author_networks']}
- **Cache Performance:** {stats['cache_size']} recommendations cached
- **Total Searches:** {stats['total_searches']}

**🕒 Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""


def create_enhanced_interface():
    """Create the enhanced Gradio interface with advanced features"""

    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
    }
    .recommendation-card {
        border: 1px solid #e1e1e1;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stats-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """

    with gr.Blocks(
        title="📚 Enhanced AI Book Recommender",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:

        gr.HTML("""
        <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 25px;">
            <h1 style="color: white; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">📚 Enhanced AI Book Recommender</h1>
            <p style="color: white; font-size: 1.3em; margin: 15px 0 0 0; opacity: 0.9;">Discover your next favorite book with advanced AI-powered recommendations</p>
            <p style="color: white; font-size: 1em; margin: 5px 0 0 0; opacity: 0.8;">🧠 Semantic Search • 🎯 Smart Filtering • 🔍 Content Analysis • 📊 Quality Scoring</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Enhanced search input
                user_input = gr.Textbox(
                    lines=4,
                    placeholder="""Enter your book search query here:
• Book titles: "The Great Gatsby", "1984"
• Authors: "Stephen King", "Jane Austen novels"
• Themes: "dystopian future", "romantic comedy", "space opera"
• Descriptions: "books like Harry Potter but darker", "mystery novels set in Victorian England"
• Moods: "feel-good stories", "mind-bending sci-fi", "emotional tear-jerkers" """,
                    label="🔍 What are you looking for?",
                    info="Be as specific as possible for better AI-powered recommendations"
                )

                # Basic controls
                with gr.Row():
                    num_recs = gr.Slider(
                        minimum=1, maximum=50, value=12, step=1,
                        label="📊 Number of Recommendations"
                    )
                    diversity_boost = gr.Checkbox(
                        value=True,
                        label="🎲 Diversity Boost",
                        info="Avoid too many similar books/authors"
                    )

                # Enhanced filtering section
                gr.Markdown("### 🎛️ **Advanced Filters & Options**")

                with gr.Accordion("📈 Quality & Popularity Filters", open=False):
                    with gr.Row():
                        min_rating = gr.Slider(
                            minimum=0.0, maximum=5.0, value=3.5, step=0.1,
                            label="⭐ Minimum Rating",
                            info="Filter by book quality"
                        )
                        min_popularity = gr.Number(
                            value=100, minimum=0, maximum=1000000,
                            label="🔥 Minimum Ratings Count",
                            info="Filter by community engagement"
                        )

                with gr.Accordion("📅 Publication & Content Filters", open=False):
                    with gr.Row():
                        year_start = gr.Number(
                            value=1950, minimum=1000, maximum=2024,
                            label="📅 From Year"
                        )
                        year_end = gr.Number(
                            value=2024, minimum=1000, maximum=2024,
                            label="📅 To Year"
                        )

                    with gr.Row():
                        genre_filter = gr.CheckboxGroup(
                            choices=[
                                "fantasy", "science_fiction", "romance", "mystery",
                                "horror", "historical", "biography", "self_help",
                                "business", "philosophy", "young_adult", "childrens"
                            ],
                            label="🎭 Include Genres (Optional)",
                            info="Select genres to focus on"
                        )

                with gr.Accordion("👤 Author & Advanced Filters", open=False):
                    author_filter = gr.Textbox(
                        label="👤 Include Authors (Optional)",
                        placeholder="e.g., 'Stephen King', 'Agatha Christie'",
                        info="Search within books by specific authors"
                    )

                    exclude_authors = gr.Textbox(
                        label="🚫 Exclude Authors (Optional)",
                        placeholder="e.g., 'Dan Brown', 'E.L. James'",
                        info="Comma-separated list of authors to exclude"
                    )

                    with gr.Row():
                        rerank_by_popularity = gr.Checkbox(
                            value=False,
                            label="📈 Rerank by Popularity",
                            info="Prioritize popular books in results"
                        )
                        include_explanations = gr.Checkbox(
                            value=True,
                            label="💡 Show Explanations",
                            info="Include why each book was recommended"
                        )

                # Enhanced action buttons
                with gr.Row():
                    recommend_btn = gr.Button(
                        "🚀 Get AI Recommendations",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                        scale=1
                    )

            with gr.Column(scale=1):
                # Enhanced dataset statistics
                stats_display = gr.Markdown(
                    get_enhanced_dataset_stats(),
                    label="Dataset Analytics"
                )

                # Advanced tips and examples
                gr.Markdown("""
### 💡 **Advanced Search Tips**

**🎯 Smart Query Strategies:**
- **Semantic:** "Books about artificial intelligence and consciousness"
- **Comparative:** "Similar to Dune but more accessible"  
- **Mood-based:** "Uplifting stories about personal growth"
- **Thematic:** "Time travel paradoxes with romance"
- **Style-based:** "Lyrical prose with magical realism"

**⚡ Pro Filter Combinations:**
- **Hidden Gems:** High rating (4.0+) + Low popularity (<1K)
- **Crowd Favorites:** High popularity (10K+) + Any rating
- **Modern Classics:** 1950-2000 + High rating (4.2+)
- **Contemporary Hits:** 2010-2024 + High popularity

**🔍 Advanced Features:**
- **Diversity Boost:** Prevents author/genre clustering
- **Popularity Rerank:** Surfaces mainstream hits
- **Genre Focus:** Combine multiple genres for precision
- **Author Networks:** Discovers similar writing styles

**🎭 Genre Combinations:**
- Fantasy + Historical = Historical fantasy
- Science Fiction + Philosophy = Philosophical sci-fi
- Mystery + Historical = Historical mysteries
                """)

                # System status
                if recommender:
                    system_stats = recommender.get_recommendation_statistics()
                    gr.Markdown(f"""
### 🤖 **AI System Status**
- **Status:** ✅ Online and Ready
- **Books Indexed:** {system_stats['total_books']:,}
- **AI Dimensions:** {system_stats['embedding_dimension']}
- **Cached Searches:** {system_stats['cache_size']}
- **Performance:** Optimized with clustering
                    """)

        # Enhanced output area with tabs
        with gr.Tabs():
            with gr.TabItem("📚 Recommendations", elem_id="recommendations-tab"):
                output = gr.Markdown(
                    value="""
# 🎯 **Ready for Your Search!**

Enter a book search query above and click "🚀 Get AI Recommendations" to discover your next great read.

**💡 Quick Start Examples:**
- "Books like The Hunger Games but for adults"
- "Carl Sagan science books"
- "Cozy mystery novels set in small towns"
- "Epic fantasy series similar to Lord of the Rings"

The AI will analyze your query and find the most relevant books using advanced semantic matching, quality scoring, and popularity signals.
                    """,
                    label="Your Personalized Recommendations"
                )

            with gr.TabItem("📊 Search Analytics", elem_id="analytics-tab"):
                analytics_output = gr.Markdown(
                    value="Search analytics will appear here after you make recommendations.",
                    label="Search Performance & Insights"
                )

        # Event handlers
        def get_enhanced_recommendations(query, num_recs, min_rat, min_pop, year_start, year_end,
                                         author_filt, exclude_auth, genre_filt, diversity, rerank,
                                         include_exp):
            if recommender is None:
                return "❌ **System Error:** Recommender not initialized. Please check if books.csv exists.", ""

            try:
                # Parse exclude authors
                exclude_list = []
                if exclude_auth:
                    exclude_list = [author.strip() for author in exclude_auth.split(
                        ',') if author.strip()]

                # Get recommendations
                year_range = (int(year_start), int(year_end))
                recommendations = recommender.recommend_books_enhanced(
                    user_input=query,
                    num_recommendations=int(num_recs),
                    min_rating=float(min_rat),
                    min_popularity=int(min_pop),
                    year_range=year_range,
                    author_filter=author_filt,
                    genre_filter=genre_filt if genre_filt else None,
                    exclude_authors=exclude_list if exclude_list else None,
                    include_explanations=include_exp,
                    diversity_boost=diversity,
                    rerank_by_popularity=rerank
                )

                # Format recommendations
                formatted_recs = format_enhanced_recommendations(
                    recommendations)

                # Generate analytics
                analytics = generate_search_analytics(
                    query, recommendations, recommender)

                return formatted_recs, analytics

            except Exception as e:
                logger.error(f"Error in interface: {e}")
                error_msg = f"❌ **Error:** {str(e)}\n\nPlease try again with different parameters."
                return error_msg, f"**Error occurred:** {str(e)}"

        def clear_all():
            return "", 12, 3.5, 100, 1950, 2024, "", "", None, True, False, True

        # Connect event handlers
        recommend_btn.click(
            fn=get_enhanced_recommendations,
            inputs=[user_input, num_recs, min_rating, min_popularity,
                    year_start, year_end, author_filter, exclude_authors,
                    genre_filter, diversity_boost, rerank_by_popularity, include_explanations],
            outputs=[output, analytics_output]
        )

        clear_btn.click(
            fn=clear_all,
            outputs=[user_input, num_recs, min_rating, min_popularity,
                     year_start, year_end, author_filter, exclude_authors,
                     genre_filter, diversity_boost, rerank_by_popularity, include_explanations]
        )

        # Enhanced examples
        gr.Markdown("### 🎯 **Try These Example Searches:**")

        enhanced_examples = [
            ["Books similar to The Martian with humor and science", 8, 4.0,
                5000, 2000, 2024, "", "", ["science_fiction"], True, False, True],
            ["Agatha Christie style mystery novels", 10, 4.0, 1000, 1920,
                1980, "Agatha Christie", "", ["mystery"], True, False, True],
            ["Epic fantasy series like Game of Thrones", 6, 4.2, 10000,
                1990, 2024, "", "", ["fantasy"], True, False, True],
            ["Self-help books about productivity and habits", 7, 4.0, 2000,
                2000, 2024, "", "", ["self_help"], True, False, True],
            ["Historical fiction set during World War II", 10, 4.0, 5000,
                1940, 2024, "", "", ["historical"], True, False, True],
            ["Romantic comedies with strong female protagonists", 8, 3.8,
                1000, 1990, 2024, "", "", ["romance"], True, False, True],
            ["Philosophy books accessible to beginners", 5, 4.0, 1000,
                1900, 2024, "", "", ["philosophy"], True, False, True],
            ["Young adult dystopian fiction", 8, 3.9, 5000, 2000, 2024, "",
                "", ["young_adult", "science_fiction"], True, False, True]
        ]

        gr.Examples(
            examples=enhanced_examples,
            inputs=[user_input, num_recs, min_rating, min_popularity,
                    year_start, year_end, author_filter, exclude_authors,
                    genre_filter, diversity_boost, rerank_by_popularity, include_explanations],
            outputs=[output, analytics_output],
            fn=get_enhanced_recommendations,
            cache_examples=False
        )

        # Enhanced footer
        gr.HTML("""
        <div style="text-align: center; padding: 25px; margin-top: 30px; border-top: 2px solid #eee; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px;">
            <h3 style="color: #333; margin-bottom: 15px;">🤖 Powered by Advanced AI Technology</h3>
            <p style="color: #666; margin: 5px 0;"><strong>🧠 AI Features:</strong> Semantic Embeddings • TF-IDF Analysis • Clustering • Quality Scoring • Diversity Boosting</p>
            <p style="color: #666; margin: 5px 0;"><strong>📊 Data Source:</strong> Goodreads Community Dataset • Machine Learning Enhanced • Real-time Processing</p>
            <p style="color: #666; margin: 5px 0;"><strong>🚀 Built with:</strong> Python • SentenceTransformers • Scikit-learn • Gradio • Advanced NLP</p>
            <hr style="margin: 20px 0; border: 1px solid #ddd;">
            <p style="color: #888; font-size: 0.9em; margin: 10px 0;">
                📚 <strong>Book Discovery:</strong> Find books on Amazon, WorldCat, or your local library<br>
                🔍 <strong>Research:</strong> Use ISBN numbers for academic citation and verification<br>
                ⚠️ <strong>Educational Use:</strong> This tool is for book discovery and research purposes. Please support authors and publishers.
            </p>
            <p style="color: #aaa; font-size: 0.8em; margin-top: 15px;">
                Made with ❤️ for book lovers everywhere • Version 2.0 Enhanced • Last Updated: {datetime.now().strftime('%Y-%m-%d')}
            </p>
        </div>
        """)

    return interface


def generate_search_analytics(query: str, recommendations: List[Dict],
                              recommender: AdvancedGoodreadsRecommender) -> str:
    """Generate detailed analytics for the search results"""
    if not recommendations or "error" in recommendations[0]:
        return "**No analytics available for this search.**"

    try:
        # Basic statistics
        total_recs = len(recommendations)
        avg_rating = sum(book.get('average_rating', 0)
                         for book in recommendations) / total_recs
        total_ratings = sum(book.get('ratings_count', 0)
                            for book in recommendations)
        avg_year = sum(book.get('publication_year', 0) for book in recommendations if isinstance(book.get('publication_year'), (int, float)) and book.get(
            'publication_year') > 0) / max(1, sum(1 for book in recommendations if isinstance(book.get('publication_year'), (int, float)) and book.get('publication_year') > 0))

        # Genre distribution
        all_genres = []
        for book in recommendations:
            if book.get('detected_genres'):
                all_genres.extend(book['detected_genres'])
        genre_counts = Counter(all_genres)
        top_genres = genre_counts.most_common(5)

        # Author diversity
        authors = set(book.get('authors', '') for book in recommendations)
        author_diversity = len(authors)

        # Decade distribution
        decades = []
        for book in recommendations:
            year = book.get('publication_year')
            if isinstance(year, (int, float)) and year > 0:
                decades.append(int(year // 10 * 10))
        decade_counts = Counter(decades)

        # Score analysis
        scores = [float(book.get('raw_score', 0))
                  for book in recommendations if book.get('raw_score')]
        if scores:
            avg_score = sum(scores) / len(scores)
            score_range = max(scores) - min(scores)
        else:
            avg_score = 0
            score_range = 0

        # System stats
        system_stats = recommender.get_recommendation_statistics()

        analytics = f"""
## 📊 **Search Analytics Dashboard**

### **🎯 Query Analysis**
- **Search Query:** "{query}"
- **Results Found:** {total_recs} books
- **Average Match Score:** {avg_score:.3f}
- **Score Distribution Range:** {score_range:.3f}

### **📈 Quality Metrics**
- **Average Rating:** {avg_rating:.2f}⭐ 
- **Total Community Ratings:** {int(total_ratings):,}
- **Average Publication Year:** {int(avg_year) if avg_year > 0 else 'Mixed'}
- **Author Diversity:** {author_diversity} unique authors ({author_diversity/total_recs*100:.1f}%)

### **🎭 Genre Distribution**
{chr(10).join([f"- **{genre.replace('_', ' ').title()}:** {count} books" for genre, count in top_genres]) if top_genres else "- Mixed genres detected"}

### **📅 Publication Timeline**
{chr(10).join([f"- **{decade}s:** {count} books" for decade, count in sorted(decade_counts.items(), reverse=True)[:5]]) if decade_counts else "- Various publication periods"}

### **🤖 AI System Performance**
- **Search Method:** Hybrid (Semantic + Keyword + Quality)
- **Processing Time:** < 1 second (cached: {system_stats['cache_size']} searches)
- **Recommendation Accuracy:** Optimized with {system_stats['book_clusters']} clusters
- **Total System Searches:** {system_stats['total_searches']}

### **💡 Search Insights**
- **Diversity Level:** {'High' if author_diversity/total_recs > 0.8 else 'Moderate' if author_diversity/total_recs > 0.5 else 'Focused'}
- **Quality Level:** {'Premium' if avg_rating >= 4.2 else 'High' if avg_rating >= 4.0 else 'Good' if avg_rating >= 3.5 else 'Mixed'}
- **Popularity Level:** {'Mainstream' if total_ratings/total_recs > 10000 else 'Popular' if total_ratings/total_recs > 1000 else 'Niche'}
- **Era Focus:** {'Contemporary' if avg_year > 2000 else 'Modern' if avg_year > 1980 else 'Classic' if avg_year > 1950 else 'Vintage'}

**🎯 Recommendation Confidence:** {'Very High' if avg_score > 0.7 else 'High' if avg_score > 0.5 else 'Good' if avg_score > 0.3 else 'Moderate'}
        """

        return analytics

    except Exception as e:
        logger.error(f"Error generating analytics: {e}")
        return f"**Analytics Error:** Could not generate detailed analytics. {str(e)}"


# Launch the enhanced application
if __name__ == "__main__":
    if recommender is not None:
        try:
            interface = create_enhanced_interface()

            # Launch with enhanced configuration
            interface.launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                show_error=True,
                show_tips=True,
                enable_queue=True,
                max_threads=10
            )

            logger.info(
                "🚀 Enhanced Goodreads Recommender launched successfully!")
            logger.info(
                "🌟 Features: Advanced AI • Smart Filtering • Analytics • Caching")

        except Exception as e:
            logger.error(f"Failed to launch interface: {e}")
            print(f"❌ Launch error: {e}")
    else:
        error_msg = """
❌ **INITIALIZATION FAILED**

The recommender system could not be initialized. Please ensure:

1. **books.csv** file exists in the current directory
2. The CSV file is properly formatted with required columns:
   - title, authors, average_rating, ratings_count, etc.
3. You have sufficient memory and disk space
4. All required Python packages are installed

For help, check the logs in 'recommender.log'
        """
        print(error_msg)
