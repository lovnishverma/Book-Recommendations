import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# Load dataset
df = pd.read_csv("books.csv")

# Combine title and author
book_texts = (df['title'] + " by " + df['authors']).fillna("").tolist()

# Load sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all book entries
embeddings = model.encode(book_texts, show_progress_bar=True)

# Book recommender function
def recommend_books(user_input):
    if not user_input.strip():
        return ["Please enter a book title or author."]
    
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    recommendations = [book_texts[i] for i in top_indices]
    return recommendations

# Gradio interface
interface = gr.Interface(
    fn=recommend_books,
    inputs=gr.Textbox(lines=1, placeholder="Type a book title or author..."),
    outputs=gr.List(label="Top 5 Recommended Books"),
    title="📚 Book Recommender",
    description="Enter a book title or author and get similar book recommendations!"
)

# Launch app
if __name__ == "__main__":
    interface.launch()
