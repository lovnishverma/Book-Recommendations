---
title: Books Recommendation
emoji: 📚
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: true
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/6474405f90330355db146c76/uCiC_ILzv0UUhGHSOBVzJ.jpeg
short_description: Enter a book title or author to get 5 similar book recommend
---


# 📚 Books Recommendation App

![Books Recommendation](https://cdn-uploads.huggingface.co/production/uploads/6474405f90330355db146c76/uCiC_ILzv0UUhGHSOBVzJ.jpeg)

**Short Description:**  
Enter a book title, author name, or description to get 5 similar book recommendations from a curated Goodreads dataset using AI-powered embeddings and hybrid search.

---

## 🚀 Live Demo

Check out the live app on Hugging Face Spaces:  
[🔗 Open in Browser](https://lovnishverma-book-recommendations.hf.space/)

---

## 🛠 Features

- Search by **book title**, **author**, or **description/theme**
- Hybrid recommendation combining:
  - **Semantic similarity** using Sentence Transformers
  - **Keyword similarity** using TF-IDF
  - **Popularity** and **rating scores**
- **Advanced filters**:
  - Minimum rating
  - Minimum ratings count
  - Publication year range
  - Author filter
- **Detailed book info** including:
  - Title, original title, author(s)
  - Publication year and decade
  - Average rating, total ratings, reviews
  - Rating distribution breakdown
  - Book cover image
  - ISBN, book ID
  - Amazon and free eBook search links
- Recommendation **explanations** (why each book was suggested)
- Fast performance using pre-computed embeddings and caching

---

## 📦 Installation / Run Locally

This app is built with **Gradio** and Python. To run locally:

1. Clone the repository:

```bash
git clone https://huggingface.co/spaces/lovnishverma/book-recommendations
cd book-recommendations
````

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place your Goodreads dataset CSV file (`books.csv`) in the project root. Make sure it has at least the following columns:

```
book_id, title, original_title, authors, average_rating, ratings_count, work_text_reviews_count, image_url, small_image_url, original_publication_year, ratings_1, ratings_2, ratings_3, ratings_4, ratings_5
```

5. Run the app:

```bash
python app.py
```

6. Open the local link in your browser (usually `http://127.0.0.1:7860/`).

---

## 🧠 How It Works

1. **Data Loading & Preprocessing**

   * Loads Goodreads dataset and cleans invalid entries.
   * Converts numeric fields and calculates additional features like popularity scores, rating distribution, and decade.

2. **Hybrid Recommendation**

   * **Semantic search:** Sentence Transformer embeddings (`all-MiniLM-L6-v2`) for content similarity.
   * **Keyword search:** TF-IDF vectorization of book titles, authors, and combined text.
   * **Scoring:** Weighted sum of semantic, keyword, popularity, rating, and rating distribution scores.

3. **Filtering**

   * User-defined filters for rating, popularity, publication year, and author.

4. **Output**

   * Returns top recommendations with detailed book info, cover image, match score, and explanation.

---

## 🎛️ Usage Tips

* Search queries can be flexible:

  * `"Harry Potter fantasy adventure"`
  * `"Stephen King horror novels"`
  * `"Science fiction with strong female protagonists"`

* Use **filters** to narrow results:

  * Minimum rating → ensures higher quality books
  * Minimum ratings count → ensures popularity
  * Year range → historical or modern preferences
  * Author filter → specific author's works

* Enable **explanations** to understand why books are recommended.

---

## 🔗 Links

* [Amazon](https://www.amazon.com/) – Quick purchase
* [Anna's Archive](https://annas-archive.org/) – Free eBook search

---

## 📊 Dataset Statistics

* Total books: Automatically computed in-app
* Unique authors: Automatically computed in-app
* Publication range: Automatically computed in-app
* Total ratings & Average rating: Automatically computed in-app
* Top rated & most popular books: Displayed in-app

---

## 📚 Tech Stack

* Python 3.x
* Gradio 5.31.0
* Pandas, NumPy
* Scikit-learn
* Sentence Transformers (`all-MiniLM-L6-v2`)
* Pickle for storing embeddings
* Logging and warnings management

---

## 📝 License

MIT License – free to use for educational and personal purposes. Please respect copyright laws and support authors.

---

## ❤️ Author / Maintainer

**Lovnish Verma**
[GitHub Profile](https://github.com/lovnishverma) | [Hugging Face Spaces](https://huggingface.co/spaces/lovnishverma)

---

## ⚠️ Disclaimer

This app is **for educational and recommendation purposes only**. eBook links should be used responsibly and legally. Support authors and purchase books whenever possible.


