from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import random
import string
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

# Set up NLTK data path
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {e}")

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///library.db'
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'
db = SQLAlchemy(app)

class Book(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    author = db.Column(db.String(100), nullable=False)
    isbn = db.Column(db.String(13), unique=True, nullable=False)

class Borrower(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    borrower_name = db.Column(db.String(100), nullable=False)
    borrow_date = db.Column(db.DateTime, nullable=False)
    return_date = db.Column(db.DateTime, nullable=False)
    unique_pass = db.Column(db.String(10), unique=True, nullable=False)
    returned = db.Column(db.Boolean, default=False)
    book = db.relationship('Book', backref='borrowers')

class Review(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    book_id = db.Column(db.Integer, db.ForeignKey('book.id'), nullable=False)
    review_text = db.Column(db.Text, nullable=False)
    sentiment_score = db.Column(db.Float)
    book = db.relationship('Book', backref='reviews')

def preprocess_text(text):
    try:
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        return [word for word in tokens if word.isalnum() and word not in stop_words]
    except LookupError:
        print("NLTK data not found. Falling back to basic tokenization.")
        return text.lower().split()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        query = request.form.get('query')
        query_tokens = preprocess_text(query)
        
        all_books = Book.query.all()
        matching_books = []
        
        for book in all_books:
            title_tokens = preprocess_text(book.title)
            author_tokens = preprocess_text(book.author)
            
            if any(token in title_tokens or token in author_tokens for token in query_tokens):
                matching_books.append(book)
        
        return render_template('search_results.html', books=matching_books)
    return render_template('search.html')

@app.route('/donate', methods=['GET', 'POST'])
def donate():
    if request.method == 'POST':
        new_book = Book(
            title=request.form.get('title'),
            author=request.form.get('author'),
            isbn=request.form.get('isbn')
        )
        db.session.add(new_book)
        db.session.commit()
        flash('Book donated successfully!', 'success')
        return redirect(url_for('home'))
    return render_template('donate.html')

@app.route('/borrow', methods=['GET', 'POST'])
def borrow():
    if request.method == 'POST':
        book_id = request.form.get('book_id')
        borrower_name = request.form.get('borrower_name')
        duration = int(request.form.get('duration'))
        
        book = Book.query.get(book_id)
        if not book:
            flash('Book not found', 'error')
            return redirect(url_for('borrow'))
        
        borrow_date = datetime.now()
        return_date = borrow_date + timedelta(days=duration)
        unique_pass = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        
        new_borrower = Borrower(
            book_id=book_id,
            borrower_name=borrower_name,
            borrow_date=borrow_date,
            return_date=return_date,
            unique_pass=unique_pass
        )
        db.session.add(new_borrower)
        db.session.commit()
        
        flash(f'Book borrowed successfully! Your unique pass is: {unique_pass}', 'success')
        return redirect(url_for('home'))
    
    books = Book.query.all()
    return render_template('borrow.html', books=books)

@app.route('/return', methods=['GET', 'POST'])
def return_book():
    if request.method == 'POST':
        title = request.form.get('title')
        author = request.form.get('author')
        unique_pass = request.form.get('unique_pass')
        
        book = Book.query.filter_by(title=title, author=author).first()
        if not book:
            flash('Book not found', 'error')
            return redirect(url_for('return_book'))
        
        borrower = Borrower.query.filter_by(book_id=book.id, unique_pass=unique_pass, returned=False).first()
        if not borrower:
            flash('Borrower record not found or book already returned', 'error')
            return redirect(url_for('return_book'))
        
        borrower.returned = True
        db.session.commit()
        flash('Book returned successfully!', 'success')
        return redirect(url_for('home'))
    
    return render_template('return.html')

@app.route('/review', methods=['GET', 'POST'])
def review():
    if request.method == 'POST':
        book_id = request.form.get('book_id')
        review_text = request.form.get('review_text')
        
        sia = SentimentIntensityAnalyzer()
        sentiment_score = sia.polarity_scores(review_text)['compound']
        
        new_review = Review(book_id=book_id, review_text=review_text, sentiment_score=sentiment_score)
        db.session.add(new_review)
        db.session.commit()
        
        flash('Review submitted successfully!', 'success')
        return redirect(url_for('home'))
    
    books = Book.query.all()
    return render_template('review.html', books=books)

@app.route('/admin')
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/admin/add_book', methods=['GET', 'POST'])
def admin_add_book():
    if request.method == 'POST':
        new_book = Book(
            title=request.form.get('title'),
            author=request.form.get('author'),
            isbn=request.form.get('isbn')
        )
        db.session.add(new_book)
        db.session.commit()
        flash('Book added successfully!', 'success')
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_add_book.html')

@app.route('/admin/reviews')
def admin_reviews():
    reviews = Review.query.all()
    return render_template('admin_reviews.html', reviews=reviews)

@app.route('/admin/borrowed_books')
def admin_borrowed_books():
    borrowed_books = Borrower.query.filter_by(returned=False).all()
    return render_template('admin_borrowed_books.html', borrowed_books=borrowed_books)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)