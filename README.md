 Review Sense Extraction (Aspect-Based Sentiment Analysis)

 Overview
Review Sense Extraction is an AI-powered web application that analyzes customer reviews to determine overall sentiment and extract aspect-level sentiments (Positive, Neutral, Negative). The system provides deeper insights into customer feedback by identifying what users like or dislike about specific product features.

---

Features
- Overall sentiment classification
- Aspect-based sentiment analysis
- Confidence scoring for predictions
- Interactive Streamlit dashboard
- Active Learning for improving model accuracy
- CSV bulk upload and real-time analysis
- Secure user authentication and workspace management

---

 Technologies Used
- Python  
- Streamlit  
- NLP: NLTK, spaCy, TextBlob  
- Machine Learning: Scikit-learn (TF-IDF, Logistic Regression)  
- Database: SQLite  
- Visualization: Plotly, Matplotlib  
- Libraries: Pandas, NumPy  

---

 Machine Learning Approach
- Text preprocessing (cleaning, tokenization, stop-word removal)
- Feature extraction using TF-IDF
- Sentiment classification using Logistic Regression
- Evaluation using Accuracy and F1-score
- Lexicon-based fallback sentiment analysis
- Active Learning using low-confidence predictions

---

How to Run the Project

 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/review-sense-extraction.git
cd review-sense-extraction

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run the Application
streamlit run app.py

Author

Sahith Bukkapatnam
B.Tech – Computer Science and Engineering
