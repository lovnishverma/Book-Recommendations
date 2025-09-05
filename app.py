
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import gradio as gr
import pickle
import os
from typing import List, Tuple, Dict, Optional
import re
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedGoodreadsRecommender:
    def __init__(self, csv_file: str = "books.csv"):
        """Initialize the advanced Goodreads book recommender system"""
        self.csv_file = csv_file
        self.df = None
        self.model = None
        self.embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.rating_scaler = MinMaxScaler()
        self.popularity_scaler = MinMaxScaler()
        self.recommendation_cache = {}
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Load or create embeddings
        self.setup_models()
    
    def load_and_preprocess_data(self):
        """Load and clean the Goodreads dataset"""
        try:
            self.df = pd.read_csv(self.csv_file)
            logger.info(f"Loaded {len(self.df)} books from Goodreads dataset")
        except FileNotFoundError:
            logger.error(f"Dataset file {self.csv_file} not found!")
            raise FileNotFoundError(f"Please ensure {self.csv_file} exists in the current directory")
        
        # Clean and process the data
        original_count = len(self.df)
        
        # Remove books without essential information
        self.df = self.df.dropna(subset=['title', 'authors'])
        self.df = self.df[self.df['title'].str.strip() != '']
        
        # Process numeric fields
        numeric_fields = ['average_rating', 'ratings_count', 'work_ratings_count', 
                         'work_text_reviews_count', 'ratings_1', 'ratings_2', 
                         'ratings_3', 'ratings_4', 'ratings_5']
        
        for field in numeric_fields:
            if field in self.df.columns:
                self.df[field] = pd.to_numeric(self.df[field], errors='coerce').fillna(0)
        
        # Process publication year
        if 'original_publication_year' in self.df.columns:
            self.df['publication_year'] = pd.to_numeric(self.df['original_publication_year'], errors='coerce')
            # Filter out unrealistic years
            self.df['publication_year'] = self.df['publication_year'].where(
                (self.df['publication_year'] >= 1000) & (self.df['publication_year'] <= 2024)
            )
        
        # Create enhanced text features
        self.df['title_clean'] = self.df['title'].apply(self.clean_text)
        self.df['authors_clean'] = self.df['authors'].apply(self.clean_text)
        self.df['original_title_clean'] = self.df.get('original_title', '').fillna('').apply(self.clean_text)
        
        # Create comprehensive text for embedding
        self.df['combined_text'] = (
            self.df['title_clean'] + " " +
            self.df['authors_clean'] + " " +
            self.df['original_title_clean']
        ).str.strip()
        
        # Calculate popularity score
        self.df['popularity_score'] = self.calculate_popularity_score()
        
        # Calculate rating distribution scores
        self.df['rating_distribution_score'] = self.calculate_rating_distribution_score()
        
        # Create decade feature
        self.df['decade'] = (self.df['publication_year'] // 10 * 10).fillna(0).astype(int)
        
        logger.info(f"Processed {len(self.df)} books (removed {original_count - len(self.df)} invalid entries)")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def calculate_popularity_score(self) -> np.ndarray:
        """Calculate popularity score based on ratings count and reviews"""
        ratings_count = self.df['ratings_count'].fillna(0)
        reviews_count = self.df['work_text_reviews_count'].fillna(0)
        popularity = np.log1p(ratings_count) * 0.7 + np.log1p(reviews_count) * 0.3
        return popularity
    
    def calculate_rating_distribution_score(self) -> np.ndarray:
        """Calculate a score based on rating distribution quality"""
        rating_cols = ['ratings_5', 'ratings_4', 'ratings_3', 'ratings_2', 'ratings_1']
        scores = []
        for idx, row in self.df.iterrows():
            ratings = [row.get(col, 0) for col in rating_cols]
            total_ratings = sum(ratings)
            if total_ratings == 0:
                scores.append(0)
                continue
            weights = [5, 4, 3, 2, 1]
            weighted_sum = sum(r * w for r, w in zip(ratings, weights))
            avg_rating = weighted_sum / total_ratings
            confidence_bonus = min(np.log1p(total_ratings) / 10, 1.0)
            final_score = avg_rating + confidence_bonus
            scores.append(final_score)
        return np.array(scores)
    
    def setup_models(self):
        """Initialize ML models and create embeddings"""
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings_file = f"{self.csv_file.replace('.csv', '')}_embeddings.pkl"
        if os.path.exists(embeddings_file):
            try:
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                logger.info("Loaded pre-computed embeddings")
                if len(self.embeddings) != len(self.df):
                    logger.warning("Embeddings size mismatch, recreating...")
                    self.create_embeddings(embeddings_file)
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
                self.create_embeddings(embeddings_file)
        else:
            self.create_embeddings(embeddings_file)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000, 
            stop_words='english', 
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.8
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])
        
        if len(self.df) > 0:
            self.rating_scaler.fit(self.df[['average_rating']])
            self.popularity_scaler.fit(self.df[['popularity_score']])
        logger.info("Models setup completed")
    
    def create_embeddings(self, save_file: str):
        """Create and save embeddings"""
        logger.info("Creating embeddings for all books...")
        book_texts = self.df['combined_text'].tolist()
        batch_size = 1000
        embeddings_list = []
        for i in range(0, len(book_texts), batch_size):
            batch = book_texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=True)
            embeddings_list.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(book_texts)-1)//batch_size + 1}")
        self.embeddings = np.vstack(embeddings_list)
        try:
            with open(save_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            logger.info(f"Embeddings saved to {save_file}")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def hybrid_search(self, query: str, num_candidates: int = 100) -> np.ndarray:
        query_embedding = self.model.encode([query])
        semantic_scores = cosine_similarity(query_embedding, self.embeddings)[0]
        query_tfidf = self.tfidf_vectorizer.transform([query])
        keyword_scores = cosine_similarity(query_tfidf, self.tfidf_matrix)[0]
        popularity_scores = self.popularity_scaler.transform(self.df[['popularity_score']]).flatten()
        rating_scores = self.rating_scaler.transform(self.df[['average_rating']]).flatten()
        combined_scores = (
            0.45 * semantic_scores +
            0.25 * keyword_scores +
            0.15 * popularity_scores +
            0.10 * rating_scores +
            0.05 * (self.df['rating_distribution_score'] / 6.0)
        )
        return combined_scores
    
    def apply_filters(self, min_rating: float = 0.0, min_ratings_count: int = 0,
                      year_range: Tuple[int, int] = (1000, 2024),
                      author_filter: str = "") -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        if min_rating > 0:
            mask &= (self.df['average_rating'] >= min_rating)
        if min_ratings_count > 0:
            mask &= (self.df['ratings_count'] >= min_ratings_count)
        if year_range[0] > 1000 or year_range[1] < 2024:
            year_mask = (
                (self.df['publication_year'] >= year_range[0]) & 
                (self.df['publication_year'] <= year_range[1])
            )
            mask &= year_mask.fillna(False)
        if author_filter.strip():
            author_mask = self.df['authors'].str.contains(
                author_filter.strip(), case=False, na=False
            )
            mask &= author_mask
        return mask
    
    def get_book_details(self, idx: int) -> Dict:
        book = self.df.iloc[idx]
        total_ratings = sum([book.get(f'ratings_{i}',0) for i in range(1,6)])
        rating_breakdown = ""
        if total_ratings > 0:
            for star in [5,4,3,2,1]:
                count = book.get(f'ratings_{star}',0)
                percentage = (count / total_ratings)*100
                rating_breakdown += f"{star}⭐: {percentage:.1f}% ({int(count):,}) "
        return {
            'book_id': book.get('book_id','N/A'),
            'title': book['title'],
            'original_title': book.get('original_title', book['title']),
            'authors': book['authors'],
            'publication_year': book.get('publication_year', 'Unknown'),
            'average_rating': book.get('average_rating',0),
            'ratings_count': int(book.get('ratings_count',0)),
            'work_text_reviews_count': int(book.get('work_text_reviews_count',0)),
            'isbn': book.get('isbn','N/A'),
            'isbn13': book.get('isbn13','N/A'),
            'image_url': book.get('image_url',''),
            'small_image_url': book.get('small_image_url',''),
            'rating_breakdown': rating_breakdown.strip(),
            'popularity_score': book.get('popularity_score',0),
            'decade': book.get('decade','Unknown')
        }
    
    def explain_recommendation(self, query: str, book_idx: int, similarity_score: float) -> str:
        book = self.df.iloc[book_idx]
        explanations = []
        query_lower = query.lower()
        if query_lower in book['title'].lower():
            explanations.append("📖 Title match")
        elif query_lower in book['authors'].lower():
            explanations.append("👤 Author match")
        if similarity_score > 0.8:
            explanations.append("🎯 Very high content similarity")
        elif similarity_score > 0.6:
            explanations.append("📚 High content similarity")
        elif similarity_score > 0.4:
            explanations.append("🔍 Good content similarity")
        if book['popularity_score'] > self.df['popularity_score'].quantile(0.9):
            explanations.append("🔥 Highly popular")
        elif book['popularity_score'] > self.df['popularity_score'].quantile(0.7):
            explanations.append("📈 Popular choice")
        if book['average_rating'] > 4.3:
            explanations.append("⭐ Excellent rating")
        elif book['average_rating'] > 4.0:
            explanations.append("⭐ Highly rated")
        if not explanations:
            explanations.append("🤖 AI content similarity")
        return " • ".join(explanations[:3])
    
    def recommend_books(self, user_input: str,
                       num_recommendations: int = 10,
                       min_rating: float = 0.0,
                       min_popularity: int = 0,
                       year_range: Tuple[int,int]=(1000,2024),
                       author_filter: str = "",
                       include_explanations: bool = True) -> List[Dict]:
        if not user_input.strip():
            return [{"error":"Please enter a search query (book title, author, or description)."}]
        cache_key = f"{user_input}_{num_recommendations}_{min_rating}_{min_popularity}_{year_range}_{author_filter}"
        if cache_key in self.recommendation_cache:
            return self.recommendation_cache[cache_key]
        try:
            filter_mask = self.apply_filters(min_rating,min_popularity,year_range,author_filter)
            if not filter_mask.any():
                return [{"error":"No books match your filters. Try relaxing the criteria."}]
            similarity_scores = self.hybrid_search(user_input)
            filtered_scores = np.where(filter_mask, similarity_scores, -1)
            top_indices = filtered_scores.argsort()[-num_recommendations*2:][::-1]
            top_indices = top_indices[filtered_scores[top_indices]>-1][:num_recommendations]
            if len(top_indices) == 0:
                return [{"error":"No similar books found. Try a different search term or adjust filters."}]
            recommendations = []
            for idx in top_indices:
                book_details = self.get_book_details(idx)
                book_details['similarity_score'] = f"{similarity_scores[idx]:.4f}"
                if include_explanations:
                    book_details['explanation'] = self.explain_recommendation(user_input, idx, similarity_scores[idx])
                recommendations.append(book_details)
            self.recommendation_cache[cache_key] = recommendations
            return recommendations
        except Exception as e:
            logger.error(f"Error in recommendation: {str(e)}")
            return [{"error": f"An error occurred: {str(e)}"}]

# Initialize recommender
try:
    recommender = AdvancedGoodreadsRecommender()
except Exception as e:
    logger.error(f"Failed to initialize recommender: {e}")
    recommender = None

def format_recommendations(recommendations: List[Dict]) -> str:
    if not recommendations:
        return "No recommendations found."
    if "error" in recommendations[0]:
        return f"❌ **Error:** {recommendations[0]['error']}"
    
    formatted = []
    for i, book in enumerate(recommendations, 1):
        amazon_query = '+'.join(book['title'].split())
        authors_query = '+'.join(book['authors'].split())
        
        # Use small_image_url if available, else fallback to image_url
        img_url = book['small_image_url'] if book['small_image_url'] else book['image_url']
        
        book_entry = f"""
## **{i}. {book['title']}**
{f"*Original: {book['original_title']}*" if book['original_title'] != book['title'] else ""}

![Book Cover]({img_url})

**👤 Author:** {book['authors']}  
**📅 Published:** {book['publication_year']} ({book['decade']}s)  
**⭐ Rating:** {book['average_rating']:.2f}/5.0  
**📊 Total Ratings:** {book['ratings_count']:,}  
**💬 Reviews:** {book['work_text_reviews_count']:,}  

**🎯 Match Score:** {book['similarity_score']}  
**💡 Why recommended:** {book.get('explanation', 'Content similarity')}

**📈 Rating Distribution:** {book['rating_breakdown']}

**📚 Book ID:** {book['book_id']} | **📖 ISBN:** {book['isbn']}

**🛒 Get This Book:**  
- [📖 Buy on Amazon](https://www.amazon.com/s?k={amazon_query}+{authors_query})  
- [📚 Free eBook Search](https://annas-archive.org/search?q={amazon_query}+{authors_query})

---
"""
        formatted.append(book_entry)
    
    return "\n".join(formatted)




def get_dataset_stats():
    """Get comprehensive dataset statistics"""
    if recommender is None or recommender.df is None:
        return "❌ Dataset not loaded"
    
    df = recommender.df
    
    # Basic stats
    total_books = len(df)
    total_authors = df['authors'].nunique()
    avg_rating = df['average_rating'].mean()
    total_ratings = df['ratings_count'].sum()
    
    # Year range
    year_min = df['publication_year'].min()
    year_max = df['publication_year'].max()
    
    # Top rated books
    top_rated = df.nlargest(3, 'average_rating')[['title', 'authors', 'average_rating']]
    
    # Most popular books
    most_popular = df.nlargest(3, 'ratings_count')[['title', 'authors', 'ratings_count']]
    
    return f"""
## 📊 **Goodreads Dataset Statistics**

**📚 Collection Overview:**
- **Total Books:** {total_books:,}
- **Unique Authors:** {total_authors:,}
- **Publication Range:** {int(year_min) if not pd.isna(year_min) else 'Unknown'} - {int(year_max) if not pd.isna(year_max) else 'Unknown'}
- **Total Ratings:** {int(total_ratings):,}
- **Average Rating:** {avg_rating:.2f}/5.0

**⭐ Highest Rated Books:**
{chr(10).join([f"• {row['title']} by {row['authors']} ({row['average_rating']:.2f}⭐)" for _, row in top_rated.iterrows()])}

**🔥 Most Popular Books:**
{chr(10).join([f"• {row['title']} by {row['authors']} ({row['ratings_count']:,} ratings)" for _, row in most_popular.iterrows()])}

**🕒 Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def create_advanced_interface():
    """Create the advanced Gradio interface"""
    
    with gr.Blocks(title="📚 Advanced Goodreads Book Recommender", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
            <h1 style="color: white; font-size: 2.5em; margin: 0;">📚 Advanced Goodreads Recommender</h1>
            <p style="color: white; font-size: 1.2em; margin: 10px 0 0 0;">Discover your next favorite book with AI-powered recommendations</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main search input
                user_input = gr.Textbox(
                    lines=3,
                    placeholder="Enter book title, author name, or describe what you're looking for...\nExamples: 'Harry Potter', 'books like 1984', 'Stephen King horror novels'",
                    label="🔍 What book are you looking for?",
                    info="Be as specific as possible for better recommendations"
                )
                
                # Basic controls
                with gr.Row():
                    num_recs = gr.Slider(
                        minimum=1, maximum=50, value=10, step=1,
                        label="📊 Number of Recommendations",
                        info="More recommendations = broader suggestions"
                    )
                
                # Advanced filters
                gr.Markdown("### 🎛️ **Advanced Filters**")
                
                with gr.Row():
                    min_rating = gr.Slider(
                        minimum=0.0, maximum=5.0, value=0.0, step=0.1,
                        label="⭐ Minimum Rating",
                        info="Filter by book quality"
                    )
                    
                    min_popularity = gr.Number(
                        value=0, minimum=0, maximum=1000000,
                        label="🔥 Minimum Ratings Count",
                        info="Filter by book popularity (0 = no limit)"
                    )
                
                with gr.Row():
                    year_start = gr.Number(
                        value=1000, minimum=1000, maximum=2024,
                        label="📅 From Year"
                    )
                    year_end = gr.Number(
                        value=2024, minimum=1000, maximum=2024,
                        label="📅 To Year"
                    )
                
                author_filter = gr.Textbox(
                    label="👤 Author Filter (Optional)",
                    placeholder="e.g., 'Stephen King', 'Agatha Christie'",
                    info="Search within books by specific authors"
                )
                
                include_explanations = gr.Checkbox(
                    value=True,
                    label="💡 Include recommendation explanations",
                    info="Show why each book was recommended"
                )
                
                # Action button
                recommend_btn = gr.Button(
                    "🚀 Get Recommendations", 
                    variant="primary",
                    size="lg"
                )
        
            with gr.Column(scale=1):
                # Dataset statistics
                stats_display = gr.Markdown(
                    get_dataset_stats(),
                    label="Dataset Info"
                )
                
                # Tips and examples
                gr.Markdown("""
### 💡 **Pro Tips:**

**🎯 Search Strategies:**
- **By Title:** "Harry Potter", "Pride and Prejudice"  
- **By Author:** "Stephen King novels", "Agatha Christie mysteries"  
- **By Genre/Theme:** "dystopian fiction", "romantic comedies"  
- **By Mood:** "dark psychological thrillers", "feel-good stories"  

**⚡ Filter Tips:**
- **High Rating + Low Popularity:** Hidden gems  
- **High Popularity + Any Rating:** Trending books  
- **Specific Years:** Find books from certain eras  
- **Author Filter:** Explore specific author's works  

**🔍 Advanced Queries:**
- "Books similar to 1984 but more modern"  
- "Fantasy novels like Lord of the Rings"  
- "Science fiction with strong female protagonists"
                """)
        
        # Output area
        output = gr.Markdown(
            label="📚 Recommendations",
            value="Enter a search query above to get personalized book recommendations!"
        )
        
        # Event handlers
        def get_recommendations(query, num_recs, min_rat, min_pop, year_start, year_end, author_filt, include_exp):
            if recommender is None:
                return "❌ **Error:** Recommender system not initialized. Please check if books.csv exists."
            
            try:
                year_range = (int(year_start), int(year_end))
                recommendations = recommender.recommend_books(
                    user_input=query,
                    num_recommendations=int(num_recs),
                    min_rating=float(min_rat),
                    min_popularity=int(min_pop),
                    year_range=year_range,
                    author_filter=author_filt,
                    include_explanations=include_exp
                )
                return format_recommendations(recommendations)
            except Exception as e:
                logger.error(f"Error in interface: {e}")
                return f"❌ **Error:** {str(e)}"
        
        recommend_btn.click(
            fn=get_recommendations,
            inputs=[user_input, num_recs, min_rating, min_popularity, 
                   year_start, year_end, author_filter, include_explanations],
            outputs=output
        )
        
        # Quick search examples
        gr.Markdown("### 🎯 **Try These Example Searches:**")
        
        examples = [
            ["Harry Potter fantasy adventure", 5, 4.0, 10000, 1990, 2010, "", True],
            ["George Orwell dystopian fiction", 8, 4.0, 5000, 1900, 2000, "", True],
            ["Agatha Christie mystery novels", 10, 3.8, 1000, 1920, 1980, "Agatha Christie", True],
            ["modern science fiction space opera", 7, 3.5, 0, 2000, 2024, "", True],
            ["romantic historical fiction", 6, 4.0, 2000, 1800, 1950, "", True]
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[user_input, num_recs, min_rating, min_popularity, 
                   year_start, year_end, author_filter, include_explanations],
            outputs=output,
            fn=get_recommendations,
            cache_examples=False
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #eee;">
            <p style="color: #666;">🤖 Powered by Advanced AI • 📊 Goodreads Dataset • 🚀 Built with Gradio</p>
            <p style="color: #888; font-size: 0.9em;">
                🛒 <strong>Book Access:</strong> Amazon links for purchasing • Anna's Archive for free eBook searches<br>
                ⚠️ <em>Educational purposes only. Respect copyright laws and support authors when possible.</em>
            </p>
        </div>
        """)
    
    return interface

# Launch the application
if __name__ == "__main__":
    if recommender is not None:
        interface = create_advanced_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
        logger.info("Advanced Goodreads Recommender launched successfully!")
    else:
        print("❌ Failed to initialize recommender. Please check if books.csv exists and is properly formatted.")