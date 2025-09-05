import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import re
import hashlib
import json
from pathlib import Path
import logging
from datetime import datetime
import gradio as gr
from urllib.parse import quote_plus
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('recommender.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedGoodreadsRecommender:
    def __init__(self, csv_file: str = "books.csv"):
        self.csv_file = csv_file
        self.df = None
        self.model = None
        self.embeddings = None
        self.reduced_embeddings = None
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.scaler = MinMaxScaler()
        self.genre_scaler = MinMaxScaler()
        self.book_networks = {}
        self.similarity_cache = {}
        self.max_cache_size = 1000
        self.cache_ttl = pd.Timedelta(minutes=30)

        # Initialize caches and stats
        self.recommendation_cache = {}
        self.search_history = []
        self.recommendation_stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'cache_hits': 0
        }
        self.author_similarity = {}

        self.genre_keywords = {
            'fantasy': ['fantasy', 'magic', 'dragon', 'wizard', 'elf', 'sword', 'quest'],
            'science_fiction': ['science fiction', 'sci-fi', 'space', 'alien', 'future', 'robot', 'cyber'],
            'mystery': ['mystery', 'detective', 'crime', 'murder', 'thriller', 'whodunit'],
            'romance': ['romance', 'love', 'relationship', 'heart', 'passion'],
            'horror': ['horror', 'ghost', 'vampire', 'zombie', 'fear', 'terror'],
            'nonfiction': ['non-fiction', 'biography', 'memoir', 'history', 'self', 'true'],
            'young_adult': ['young adult', 'teen', 'high school', 'coming of age'],
            'childrens': ['children', 'kids', 'picture book', 'bedtime'],
            'historical': ['historical', 'war', 'period', 'medieval', 'ancient'],
            'dystopian': ['dystopia', 'apocalypse', 'post-apocalyptic', 'totalitarian'],
            'philosophy': ['philosophy', 'wisdom', 'ethics', 'meaning'],
            'self_help': ['self help', 'motivation', 'success', 'habits', 'improvement'],
            'business': ['business', 'management', 'leadership', 'entrepreneur']
        }

    def load_and_preprocess_data(self):
        """Enhanced data loading with better error handling and preprocessing"""
        try:
            encodings = ['utf-8', 'latin1', 'iso-8859-1']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(self.csv_file, encoding=enc)
                    logger.info(f"Loaded CSV with encoding: {enc}")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                raise ValueError("Could not decode file with any encoding")

            self.df = df.copy()

            # Clean column names
            self.df.columns = [col.strip().lower().replace(' ', '_') for col in self.df.columns]

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
                'publication_year': (1000, 2025),
                'num_pages': (1, 10000),
                'ratings_1': (0, np.inf),
                'ratings_2': (0, np.inf),
                'ratings_3': (0, np.inf),
                'ratings_4': (0, np.inf),
                'ratings_5': (0, np.inf),
                'books_count': (1, np.inf)
            }

            for field, (min_val, max_val) in numeric_fields.items():
                if field in self.df.columns:
                    self.df[field] = pd.to_numeric(self.df[field], errors='coerce')
                    self.df = self.df[(self.df[field] >= min_val) & (self.df[field] <= max_val)]
                    self.df[field] = self.df[field].fillna(0)

            # Create title-author hash for deduplication
            self.df['title_author_hash'] = (self.df['title'].astype(str) + '_' + self.df['authors'].astype(str)).apply(
                lambda x: hashlib.md5(x.encode()).hexdigest())
            duplicates = self.df.duplicated(subset=['title_author_hash'], keep='first')
            self.df = self.df[~duplicates].copy()

            logger.info(f"Processed {len(self.df)} books (removed {original_count - len(self.df)} invalid/duplicate entries)")

        except FileNotFoundError:
            logger.error(f"Dataset file {self.csv_file} not found")
            raise FileNotFoundError(f"Please ensure {self.csv_file} exists in the current directory")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better normalization"""
        if pd.isna(text) or text == '':
            return ""
        text = str(text).lower()
        text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        text = re.sub(r'[^\w\s\-\']', ' ', text)  # Keep hyphens and apostrophes
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "'re": " are", "'ve": " have", "'ll": " will", "'d": " would"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _create_weighted_text(self) -> pd.Series:
        """Create weighted text combination for better embeddings"""
        weighted_parts = []
        self.df['title_clean'] = self.df['title'].apply(self.clean_text)
        self.df['authors_clean'] = self.df['authors'].apply(self.clean_text)

        # Title gets highest weight (3x)
        title_weighted = (self.df['title_clean'] + ' ') * 3
        weighted_parts.append(title_weighted)

        # Authors (2x)
        authors_weighted = (self.df['authors_clean'] + ' ') * 2
        weighted_parts.append(authors_weighted)

        # Description/summary if available
        if 'description' in self.df.columns:
            self.df['description_clean'] = self.df['description'].apply(self.clean_text)
            weighted_parts.append(self.df['description_clean'])

        # Combine all parts
        return pd.Series([' '.join(parts) for parts in zip(*weighted_parts)])

    def setup_models(self):
        """Initialize and setup all ML models"""
        try:
            logger.info("Loading AI models...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.setup_tfidf()
            self.setup_dimensionality_reduction()
            self.fit_scalers()
            logger.info("Enhanced models setup completed")
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            raise

    def setup_tfidf(self):
        """Enhanced TF-IDF setup with optimized parameters"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=15000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.7,
            sublinear_tf=True,
            norm='l2'
        )
        combined_text = self._create_weighted_text()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        self.tfidf_embeddings = tfidf_matrix.toarray()

    def setup_dimensionality_reduction(self):
        """Setup SVD for faster semantic search"""
        self.svd_model = TruncatedSVD(n_components=128, random_state=42)
        combined_text = self._create_weighted_text()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
        self.reduced_embeddings = self.svd_model.fit_transform(tfidf_matrix)

    def fit_scalers(self):
        """Fit scalers for feature normalization"""
        if 'average_rating' in self.df.columns:
            self.scaler.fit(self.df[['average_rating', 'ratings_count']].values)
        if 'decade' in self.df.columns:
            self.df['decade'] = (self.df['publication_year'] // 10 * 10).astype(int)
        else:
            self.df['decade'] = 2000

    def create_embeddings(self, output_file: Path, metadata_file: Path):
        """Create and save embeddings"""
        try:
            logger.info("Creating semantic embeddings...")
            combined_text = self._create_weighted_text()
            self.embeddings = self.model.encode(combined_text.tolist(), show_progress_bar=True)
            np.save(output_file, self.embeddings)
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'n_books': len(self.df),
                'model': 'all-MiniLM-L6-v2'
            }
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)
            logger.info(f"Embeddings saved to {output_file}")
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")

    def load_embeddings(self, embeddings_file: Path):
        """Load pre-computed embeddings"""
        try:
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings from {embeddings_file}")
            else:
                logger.warning(f"Embeddings file not found: {embeddings_file}")
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")

    def _should_recreate_embeddings(self, embeddings_file: Path, metadata_file: Path) -> bool:
        """Check if embeddings need to be recreated"""
        if not embeddings_file.exists() or not metadata_file.exists():
            return True
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            current_hash = hashlib.md5(str(len(self.df)).encode()).hexdigest()
            return metadata.get('n_books', 0) != len(self.df)
        except:
            return True

    def build_book_networks(self):
        """Build network of similar books for recommendations"""
        logger.info("Building book similarity networks...")
        self.df['genre_scores'] = self._detect_genres_batch()
        for idx in self.df.index:
            similar_books = []
            current_book = self.df.iloc[idx]

            # Find books by same author
            same_author_mask = self.df['authors'] == current_book['authors']
            same_author_books = self.df[same_author_mask].index.tolist()
            similar_books.extend(same_author_books[:5])

            # Find books in same genre
            if 'genre_scores' in self.df.columns:
                genre_mask = self.df['genre_scores'].apply(
                    lambda x: any(sim > 0.3 for sim in x.values()) if isinstance(x, dict) else False)
                similar_genre_books = self.df[genre_mask].index.tolist()
                similar_books.extend(similar_genre_books[:5])

            self.book_networks[idx] = list(set(similar_books) - {idx})[:10]

    def _detect_genres_batch(self) -> pd.Series:
        """Batch detect genres using keyword matching"""
        def detect_genres(row):
            text = f"{row['title']} {row['authors']}".lower()
            scores = {}
            for genre, keywords in self.genre_keywords.items():
                score = sum(1 for kw in keywords if kw in text) / len(keywords)
                if score > 0:
                    scores[genre] = score
            return scores

        return self.df.apply(detect_genres, axis=1)

    def apply_filters(self, min_rating: float = 0.0, min_popularity: float = 0.0,
                      year_start: int = 1000, year_end: int = 2025,
                      language_filter: str = "", max_pages: int = None,
                      author_filter: str = "", exclude_authors: List[str] = None) -> np.ndarray:
        """Apply comprehensive filters to book dataset"""
        try:
            mask = np.ones(len(self.df), dtype=bool)

            if 'average_rating' in self.df.columns:
                mask &= self.df['average_rating'] >= min_rating

            if 'ratings_count' in self.df.columns:
                popularity_mask = self.df['ratings_count'] >= min_popularity
                mask &= popularity_mask

            if 'publication_year' in self.df.columns:
                year_mask = ((self.df['publication_year'] >= year_start) &
                           (self.df['publication_year'] <= year_end))
                mask &= year_mask

            if author_filter:
                author_mask = self.df['authors'].str.contains(author_filter, case=False, na=False)
                mask &= author_mask

            if exclude_authors:
                for author in exclude_authors:
                    exclude_mask = ~self.df['authors'].str.contains(author, case=False, na=False)
                    mask &= exclude_mask

            if language_filter:
                language_mask = self.df['language_code'].str.contains(language_filter, case=False, na=False)
                mask &= language_mask

            if max_pages and 'num_pages' in self.df.columns:
                page_mask = ((self.df['num_pages'] <= max_pages) | (self.df['num_pages'].isna()))
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
            rating_cols = ['ratings_5', 'ratings_4', 'ratings_3', 'ratings_2', 'ratings_1']
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

            # Generate purchase links
            title = book['title'].strip()
            author = book['authors'].strip()
            query = quote_plus(f"{title} {author}")
            amazon_url = f"https://www.amazon.com/s?k={query}"
            ebook_url = f"https://annas-archive.org/search?q={query}"

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
                'purchase_links': {
                    'amazon': amazon_url,
                    'ebook_search': ebook_url
                }
            }
        except Exception as e:
            logger.error(f"Error getting book details for index {idx}: {e}")
            return {'error': f'Could not retrieve details for book at index {idx}'}

    def enhanced_similarity_search(self, query: str, num_candidates: int = 200) -> Tuple[np.ndarray, Dict]:
        """Enhanced similarity search with multiple algorithms and caching"""
        cache_key = hashlib.md5(f"{query}_{num_candidates}".encode()).hexdigest()
        if cache_key in self.similarity_cache:
            cached_result, timestamp = self.similarity_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_result

        final_scores = np.zeros(len(self.df))
        score_components = {}

        try:
            # 1. Semantic similarity
            if self.embeddings is not None:
                query_embedding = self.model.encode([query], normalize_embeddings=True)
                if self.reduced_embeddings is not None:
                    query_reduced = self.svd_model.transform(query_embedding)
                    semantic_scores = cosine_similarity(query_reduced, self.reduced_embeddings)[0]
                else:
                    semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]
                score_components['semantic'] = semantic_scores
                final_scores += 0.40 * semantic_scores

            # 2. TF-IDF similarity
            query_vec = self.tfidf_vectorizer.transform([self.clean_text(query)])
            if self.reduced_embeddings is not None:
                query_tfidf_reduced = self.svd_model.transform(query_vec)
                tfidf_scores = cosine_similarity(query_tfidf_reduced, self.reduced_embeddings)[0]
            else:
                tfidf_scores = cosine_similarity(query_vec, self.tfidf_embeddings)[0]
            score_components['tfidf'] = tfidf_scores
            final_scores += 0.30 * tfidf_scores

            # 3. Rating quality
            if 'average_rating' in self.df.columns and 'ratings_count' in self.df.columns:
                rating_quality = self.scaler.transform(
                    self.df[['average_rating', 'ratings_count']].values
                )[:, 0]
                score_components['rating_quality'] = rating_quality
                final_scores += 0.15 * rating_quality

            # 4. Genre match
            query_lower = query.lower()
            genre_scores = np.zeros(len(self.df))
            for idx, row in self.df.iterrows():
                text = f"{row['title']} {row['authors']}".lower()
                score = sum(1 for kw in ['fantasy', 'sci-fi', 'mystery'] if kw in query_lower and kw in text)
                genre_scores[idx] = min(score, 1.0)
            score_components['genre_match'] = genre_scores
            final_scores += 0.15 * genre_scores

            # Cache result
            result = (final_scores, score_components)
            self.similarity_cache[cache_key] = (result, datetime.now())
            if len(self.similarity_cache) > self.max_cache_size:
                oldest_key = min(self.similarity_cache.keys(), key=lambda k: self.similarity_cache[k][1])
                del self.similarity_cache[oldest_key]
            return result

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return np.zeros(len(self.df)), {}

    def _apply_diversity_boosting(self, candidate_indices: np.ndarray, target_count: int) -> np.ndarray:
        """Apply diversity boosting to avoid too many similar books"""
        try:
            selected_indices = []
            candidate_books = self.df.iloc[candidate_indices]
            selected_authors = set()
            selected_decades = set()
            selected_genres = set()

            if len(candidate_indices) > 0:
                selected_indices.append(candidate_indices[0])

            for idx in candidate_indices[1:]:
                if len(selected_indices) >= target_count:
                    break
                book = self.df.iloc[idx]
                author = book['authors']
                decade = book['decade']
                genres = book.get('detected_genres', [])

                if (author not in selected_authors or
                    decade not in selected_decades or
                    any(g not in selected_genres for g in genres)):
                    selected_indices.append(idx)
                    selected_authors.add(author)
                    selected_decades.add(decade)
                    selected_genres.update(genres)

            while len(selected_indices) < target_count and len(selected_indices) < len(candidate_indices):
                for idx in candidate_indices:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                        break

            return np.array(selected_indices[:target_count])
        except Exception as e:
            logger.error(f"Error in diversity boosting: {e}")
            return candidate_indices[:target_count]

    def explain_recommendation(self, idx: int, score_components: Dict, similarity_score: float) -> str:
        """Generate natural language explanation for recommendation"""
        try:
            explanations = []
            book = self.df.iloc[idx]

            if 'semantic' in score_components and score_components['semantic'][idx] > 0.7:
                explanations.append("Excellent semantic match")

            if book.get('average_rating', 0) > 4.3:
                explanations.append("Outstanding rating (4.4+)")

            if 'genre_match' in score_components and score_components['genre_match'][idx] > 0.5:
                genre = next((g for g in book.get('detected_genres', []) if g in self.genre_keywords), 'genre')
                explanations.append(f"{genre.replace('_', ' ').title()} genre match")

            if book.get('ratings_count', 0) > 10000:
                explanations.append("Highly popular book")

            top_explanations = explanations[:2] or ["Content similarity"]
            return f"{' • '.join(top_explanations)} (Score: {similarity_score:.3f})"
        except Exception as e:
            logger.error(f"Error explaining recommendation: {e}")
            return f"Content similarity (Score: {similarity_score:.3f})"

    def get_recommendation_statistics(self) -> Dict:
        """Get detailed statistics about the recommendation system"""
        return {
            'total_books': len(self.df) if self.df is not None else 0,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'cache_size': len(self.recommendation_cache),
            'similarity_cache_size': len(self.similarity_cache),
            'total_searches': self.recommendation_stats.get('total_searches', 0),
            'successful_searches': self.recommendation_stats.get('successful_searches', 0),
            'failed_searches': self.recommendation_stats.get('failed_searches', 0),
            'cache_hits': self.recommendation_stats.get('cache_hits', 0),
            'book_networks': len(self.book_networks),
            'author_networks': len(self.author_similarity)
        }


# Global recommender instance
recommender = None


def format_enhanced_recommendations(recommendations: List[Dict], show_debug_info: bool = False) -> str:
    """Format recommendations with enhanced display"""
    if not recommendations:
        return "No recommendations found."

    if "error" in recommendations[0]:
        return f"❌ **Error:** {recommendations[0]['error']}"

    formatted_parts = []

    # Header
    total_books = len(recommendations)
    avg_rating = sum(book.get('average_rating', 0) for book in recommendations) / total_books
    total_ratings = sum(book.get('ratings_count', 0) for book in recommendations)

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

        book_section = f"""
### **{book.get('rank', '?')}. {book['title']}**
{f"*({book['original_title']})*" if book.get('original_title') != book['title'] else ""}

**👤 Author:** {book['authors']}  
**📅 Published:** {book['publication_year']} • **📚 Series books:** {book.get('books_count', 1)}  
**⭐ Rating:** {book['average_rating']:.2f}/5.0 **({book['ratings_count']:,} ratings, {book['work_text_reviews_count']:,} reviews)**  

**🎯 Match Score:** {book['similarity_score']:.4f} • **🏆 Quality Score:** {book.get('rating_quality_score', 0):.2f}  
**💡 Why recommended:** {book.get('explanation', 'Content similarity')}

"""

        if book.get('rating_breakdown'):
            book_section += f"**📊 Rating Distribution:** {book['rating_breakdown']}\n\n"

        if book.get('detected_genres'):
            genres_text = ' • '.join([g.replace('_', ' ').title() for g in book['detected_genres'][:3]])
            book_section += f"**🎭 Genres:** {genres_text}\n\n"

        metadata_parts = []
        if book.get('isbn13') != 'N/A':
            metadata_parts.append(f"ISBN13: {book['isbn13']}")
        if book.get('num_pages') != 'Unknown':
            metadata_parts.append(f"{book['num_pages']} pages")
        if book.get('language_code') != 'Unknown':
            metadata_parts.append(f"Language: {book['language_code']}")

        if metadata_parts:
            book_section += f"**📖 Details:** {' • '.join(metadata_parts)}\n\n"

        # 🛒 Get This Book Section
        if book.get('purchase_links'):
            amazon_link = book['purchase_links']['amazon']
            ebook_link = book['purchase_links']['ebook_search']
            book_section += f"""**🛒 Get This Book:**  
- [📖 Buy on Amazon]({amazon_link}) — Purchase physical/digital copy  
- [📚 Free eBook Search]({ebook_link}) — Find free digital version

"""

        if book.get('similar_books'):
            similar_list = [f"{b['title']} ({b['rating']:.1f}⭐)" for b in book['similar_books'][:2]]
            if similar_list:
                book_section += f"**🔗 Similar books:** {' • '.join(similar_list)}\n\n"

        if show_debug_info and book.get('score_breakdown'):
            breakdown = book['score_breakdown']
            debug_info = [f"{k}: {v:.3f}" for k, v in breakdown.items()]
            book_section += f"**🔍 Score breakdown:** {' • '.join(debug_info)}\n\n"

        book_section += "---\n"
        formatted_parts.append(book_section)

    return "\n".join(formatted_parts)


def generate_search_analytics(query: str, recommendations: List[Dict], recommender) -> str:
    """Generate analytics for the search"""
    if not recommendations or 'error' in recommendations[0]:
        return "**No analytics available**"

    total_books = len(recommendations)
    avg_rating = np.mean([b.get('average_rating', 0) for b in recommendations])
    languages = [b.get('language_code', 'Unknown') for b in recommendations]
    top_lang = max(set(languages), key=languages.count) if languages else 'Unknown'

    return f"""
### 📊 **Search Analytics**
- **Query:** "{query}"
- **Results:** {total_books} books
- **Avg Rating:** {avg_rating:.2f}⭐
- **Dominant Language:** {top_lang}
- **Freshness:** {len([b for b in recommendations if b.get('publication_year', 0) > 2010])} books from last 10 years
"""


def get_enhanced_recommendations(user_input: str, num_recommendations: int = 12,
                                 min_rating: float = 3.5, min_popularity: int = 100,
                                 year_start: int = 1950, year_end: int = 2024,
                                 author_filter: str = "", exclude_authors: str = "",
                                 include_explanations: bool = True, diversity: bool = True,
                                 rerank: bool = False) -> Tuple[str, str]:
    """Enhanced recommendation function with comprehensive filtering"""
    global recommender
    try:
        if recommender is None:
            return "❌ **Error:** Recommender not initialized. Please check logs.", ""

        exclude_list = [a.strip() for a in exclude_authors.split(',') if a.strip()] if exclude_authors else None

        filter_mask = recommender.apply_filters(
            min_rating=min_rating, min_popularity=min_popularity,
            year_start=year_start, year_end=year_end,
            author_filter=author_filter, exclude_authors=exclude_list
        )

        if not filter_mask.any():
            return "❌ **No books match your filters. Try relaxing the criteria.**", ""

        similarity_scores, score_components = recommender.enhanced_similarity_search(user_input, num_candidates=min(1000, len(recommender.df)))
        filtered_scores = np.where(filter_mask, similarity_scores, -1)
        top_indices = filtered_scores.argsort()[-(num_recommendations * (3 if diversity else 2)):][::-1]
        top_indices = top_indices[filtered_scores[top_indices] > -1]

        if len(top_indices) == 0:
            return "❌ **No similar books found. Try a different search term or adjust filters.**", ""

        if diversity and len(top_indices) > num_recommendations:
            top_indices = recommender._apply_diversity_boosting(top_indices, num_recommendations)
        else:
            top_indices = top_indices[:num_recommendations]

        if rerank and 'popularity_score' in recommender.df.columns:
            pop_scores = recommender.df.iloc[top_indices]['popularity_score'].values
            top_indices = top_indices[np.argsort(-pop_scores)]

        recommendations = []
        for i, idx in enumerate(top_indices):
            book_details = recommender.get_enhanced_book_details(idx)
            book_details['rank'] = i + 1
            book_details['similarity_score'] = similarity_scores[idx]
            if include_explanations:
                book_details['explanation'] = recommender.explain_recommendation(
                    idx, score_components, similarity_scores[idx]
                )
            book_details['score_breakdown'] = {k: float(v[idx]) for k, v in score_components.items()}
            recommendations.append(book_details)

        formatted_recs = format_enhanced_recommendations(recommendations)
        analytics = generate_search_analytics(user_input, recommendations, recommender)

        return formatted_recs, analytics

    except Exception as e:
        logger.error(f"Error in interface: {e}")
        return f"❌ **Error:** {str(e)}\n\nPlease try again with different parameters.", ""


def clear_all():
    return "", 12, 3.5, 100, 1950, 2024, "", "", None, True, False, True


# Initialize recommender
try:
    recommender = AdvancedGoodreadsRecommender("books.csv")
    recommender.load_and_preprocess_data()
    recommender.setup_models()

    embeddings_file = Path("book_embeddings.npy")
    metadata_file = Path("embeddings_metadata.json")

    if recommender._should_recreate_embeddings(embeddings_file, metadata_file):
        recommender.create_embeddings(embeddings_file, metadata_file)
    else:
        recommender.load_embeddings(embeddings_file)

    recommender.build_book_networks()
    logger.info("System initialization completed successfully")
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    print(f"❌ Initialization failed: {e}")


# Gradio Interface
with gr.Blocks(title="📚 AI Book Recommender", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 📚 AI-Powered Book Recommender
    Discover your next great read with advanced AI that understands your taste!

    Enter any book, author, theme, or mood to get personalized recommendations.
    """)

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                placeholder="""Enter your book search query here:
• Book titles: "The Great Gatsby", "1984"
• Authors: "Stephen King", "Jane Austen novels"
• Themes: "dystopian future", "romantic comedy", "space opera"
• Descriptions: "books like Harry Potter but darker"
• Moods: "feel-good stories", "mind-bending sci-fi"
                """,
                label="🔍 What are you looking for?",
                lines=3
            )
        with gr.Column(scale=1):
            recommend_btn = gr.Button("🎯 Get Recommendations", variant="primary")

    with gr.Row():
        num_recs = gr.Slider(minimum=1, maximum=50, value=12, step=1, label="Number of Recommendations")
        min_rating = gr.Slider(minimum=0.0, maximum=5.0, value=3.5, step=0.1, label="Minimum Rating")
        min_popularity = gr.Slider(minimum=0, maximum=10000, value=100, step=10, label="Min Ratings Count")

    with gr.Row():
        year_start = gr.Slider(minimum=1000, maximum=2024, value=1950, step=1, label="Publication Year From")
        year_end = gr.Slider(minimum=1000, maximum=2024, value=2024, step=1, label="To")
        author_filter = gr.Textbox(label="Author Filter (optional)")
        exclude_authors = gr.Textbox(label="Exclude Authors (comma-separated)")

    with gr.Row():
        include_exp = gr.Checkbox(value=True, label="Include Explanations")
        diversity = gr.Checkbox(value=True, label="Diversity Boost")
        rerank = gr.Checkbox(value=False, label="Rerank by Popularity")

    with gr.Row():
        clear_btn = gr.Button("🗑️ Clear All")
        info_btn = gr.Button("ℹ️ How It Works")

    with gr.Tabs():
        with gr.Tab("Recommendations"):
            output = gr.Markdown()
        with gr.Tab("Analytics"):
            analytics = gr.Markdown()

    # Info modal
    with gr.Accordion("ℹ️ How This Recommender Works", visible=False) as info_box:
        gr.Markdown("""
        ### 🤖 How Our AI Book Recommender Works

        This system combines **semantic search**, **content analysis**, and **user preferences** to give smart, personalized book suggestions.

        ### 🔍 Key Features
        - **Semantic Matching**: Understands meaning, not just keywords
        - **Quality Filtering**: Prioritizes well-rated, popular books
        - **Diversity**: Recommends varied authors and genres
        - **Real-Time Controls**: Filter by year, rating, author, and more

        ### 💡 Tips for Best Results
        - Try: _"Books like The Martian with humor and science"_
        - Try: _"Dark fantasy with dragons and political intrigue"_
        - Try: _"Feel-good romance set in a small town"_
        """)

    # Connect event handlers
    recommend_btn.click(
        fn=get_enhanced_recommendations,
        inputs=[user_input, num_recs, min_rating, min_popularity,
                year_start, year_end, author_filter, exclude_authors,
                include_exp, diversity, rerank],
        outputs=[output, analytics]
    )

    clear_btn.click(fn=clear_all, outputs=[
        user_input, num_recs, min_rating, min_popularity,
        year_start, year_end, author_filter, exclude_authors,
        include_exp, diversity, rerank, analytics
    ])

    info_btn.click(fn=lambda: gr.update(visible=True), outputs=info_box)

    # System status
    if recommender and hasattr(recommender, 'df'):
        system_stats = recommender.get_recommendation_statistics()
        gr.Markdown(f"""
        ### 🤖 **AI System Status**
        - **Status:** ✅ Online and Ready
        - **Books Indexed:** {system_stats['total_books']:,}
        - **AI Dimensions:** {system_stats['embedding_dimension']}
        """)

if __name__ == "__main__":
    app.launch()