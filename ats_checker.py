import sys
import os
import re
import docx
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import ssl # Import the ssl module

# --- TEMPORARY SSL VERIFICATION DISABLE (Solution 4) ---
# This block temporarily disables SSL certificate verification for NLTK downloads.
# Use with caution and only if you understand the security implications.
# It's recommended to remove this once the underlying SSL issue is resolved.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't have _create_unverified_context
    pass
else:
    # Handle target environment that doesn't have certificate verification
    ssl._create_default_https_context = _create_unverified_https_context
# --- END OF TEMPORARY SSL VERIFICATION DISABLE ---


# --- Download NLTK data if not already present ---
# This is a robust way to ensure the necessary NLTK data is available.
try:
    stopwords.words('english')
except LookupError:
    print("NLTK 'stopwords' not found. Downloading...")
    nltk.download('stopwords', quiet=True)

# The 'punkt' tokenizer is required for word_tokenize.
# We test if it's available by tokenizing a sample sentence.
# If it fails with a LookupError, we download the necessary data.
try:
    word_tokenize("test")
except LookupError:
    print("NLTK 'punkt' tokenizer not found or is incomplete. Downloading...")
    nltk.download('punkt', quiet=True)

# --- Function to Extract Text from Different File Formats ---

def extract_text(file_path):
    """
    Extracts text from a file (.pdf, .docx, .txt).

    Args:
        file_path (str): The path to the file.

    Returns:
        str: The extracted text from the file, or an empty string if the format
             is not supported or an error occurs.
    """
    text = ""
    file_extension = os.path.splitext(file_path)[1].lower()

    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            print(f"Unsupported file format: {file_extension}")
            return ""
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except Exception as e:
        print(f"An error occurred while reading {file_path}: {e}")
        return ""
        
    return text

# --- Function to Preprocess Text ---

def preprocess_text(text):
    """
    Cleans and preprocesses the input text.
    - Converts to lowercase
    - Removes punctuation and special characters
    - Tokenizes text
    - Removes stop words

    Args:
        text (str): The raw text to be processed.

    Returns:
        str: The cleaned and processed text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Tokenize the text (split into words)
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    return " ".join(filtered_tokens)

# --- Main Function to Calculate Similarity ---

def get_ats_score(resume_path, job_description_path):
    """
    Calculates the ATS match score between a resume and a job description.

    Args:
        resume_path (str): Path to the resume file.
        job_description_path (str): Path to the job description file.
    """
    # Step 1: Extract text from files
    resume_text = extract_text(resume_path)
    jd_text = extract_text(job_description_path)

    if not resume_text or not jd_text:
        print("Could not proceed due to an error in file reading.")
        return

    # Step 2: Preprocess the extracted text
    processed_resume = preprocess_text(resume_text)
    processed_jd = preprocess_text(jd_text)
    
    # Step 3: Calculate Cosine Similarity
    text_corpus = [processed_resume, processed_jd]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_corpus)
    
    # Calculate the cosine similarity
    # The result is a matrix, we need the value at [0, 1]
    similarity_matrix = cosine_similarity(tfidf_matrix)
    similarity_score = similarity_matrix[0, 1]
    
    # Print the result
    print("\n--- ATS Match Report ---")
    print(f"Resume: {os.path.basename(resume_path)}")
    print(f"Job Description: {os.path.basename(job_description_path)}")
    print("-" * 26)
    print(f"Match Score: {similarity_score:.2%}")
    print("------------------------")
    
    if similarity_score < 0.3:
        print("Recommendation: Poor match. Consider significant revisions.")
    elif similarity_score < 0.6:
        print("Recommendation: Moderate match. Tailor your resume with more keywords from the job description.")
    else:
        print("Recommendation: Good match! Your resume aligns well with the job description.")


# --- How to Use ---
if __name__ == "__main__":


    args = sys.argv[1:]
    resume_file = args[0] 

    # --- DEFINE YOUR FILE PATHS HERE ---
    job_desc_file = "job_description/data_engineer_description.txt" # Change to the job description's filename
    
    # Check if the placeholder files exist before running
    if not os.path.exists(resume_file) or not os.path.exists(job_desc_file):
        print("\n--- SETUP REQUIRED ---")
        print(f"Error: Make sure the files '{resume_file}' and '{job_desc_file}' exist in the same directory as the script.")
        print("Please update the 'resume_file' and 'job_desc_file' variables in the script with your filenames.")
        print("----------------------\n")
    else:
        get_ats_score(resume_file, job_desc_file)
