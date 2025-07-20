import re
import os
import io # New: To capture print output
import sys # New: To redirect stdout
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from collections import Counter
import PyPDF2
from docx import Document

# NLTK for stop words
import nltk
from nltk.corpus import stopwords
# Download stopwords quietly if not present
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('english'))

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() or ""
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return ""
    return text

def extract_text_from_docx(docx_path: str) -> str:
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(docx_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX file {docx_path}: {e}")
        return ""
    return text

def extract_text_from_txt(txt_path: str) -> str:
    """Extracts text from a TXT file."""
    text = ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except Exception as e:
        print(f"Error reading TXT file {txt_path}: {e}")
        return ""
    return text

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text from a given file path based on its extension.
    Supports .txt, .pdf, and .docx files.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return ""

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    else:
        print(f"Unsupported file type: {file_extension}. Please provide a .txt, .pdf, or .docx file.")
        return ""

def _get_meaningful_tokens(keywords_list: list) -> set:
    """
    Takes a list of keywords/keyphrases, breaks them into individual words,
    removes stop words, and returns a set of unique, meaningful tokens.
    """
    tokens = set()
    for phrase in keywords_list:
        # Split by spaces and non-alphanumeric characters
        words = re.findall(r'\b\w+\b', phrase.lower())
        for word in words:
            if word not in STOP_WORDS and len(word) > 1: # Exclude single-character words and stop words
                tokens.add(word)
    return tokens

def extract_keywords_keybert(
    text: str,
    num_keywords: int = 15,
    keyphrase_ngram_range: tuple = (1, 3),
    diversity: float = 0.7,
    model: str = 'all-MiniLM-L6-v2'
) -> list:
    """
    Extracts keywords and keyphrases from a given text using KeyBERT.
    """
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    cleaned_text = cleaned_text.lower()

    if not cleaned_text.strip():
        # print("Warning: Cleaned text is empty. Cannot extract keywords.") # Suppress this warning during file processing
        return []

    try:
        kw_model = KeyBERT(model=model)
    except Exception as e:
        print(f"Error loading model {model}: {e}", file=sys.stderr) # Print to stderr for errors
        print("Attempting to load a common fallback model: 'all-MiniLM-L6-v2'", file=sys.stderr)
        try:
            kw_model = KeyBERT(model='all-MiniLM-L6-v2')
        except Exception as fallback_e:
            print(f"Failed to load fallback model: {fallback_e}", file=sys.stderr)
            return []

    keywords_with_scores = kw_model.extract_keywords(
        cleaned_text,
        keyphrase_ngram_range=keyphrase_ngram_range,
        stop_words='english',
        top_n=num_keywords,
        use_mmr=True,
        diversity=diversity
    )

    keywords = [kw for kw, score in keywords_with_scores]
    return keywords

def calculate_ats_score_and_report(resume_text: str, jd_text: str, output_stream=sys.stdout) -> dict:
    """
    Calculates a more robust ATS score and generates a detailed report on keywords.
    Prints output to the specified output_stream (defaults to stdout).
    """
    if not jd_text.strip():
        print("Job description text is empty. Cannot perform analysis.", file=output_stream)
        return {
            "ats_score": 0,
            "matched_phrases": [],
            "missing_phrases": [],
            "missing_core_tokens": [],
            "jd_extracted_keywords": [],
            "resume_extracted_keywords": []
        }
    if not resume_text.strip():
        print("Resume text is empty. Cannot perform analysis.", file=output_stream)
        return {
            "ats_score": 0,
            "matched_phrases": [],
            "missing_phrases": [],
            "missing_core_tokens": [],
            "jd_extracted_keywords": [],
            "resume_extracted_keywords": []
        }

    print("\n--- Extracting Keywords from Job Description ---", file=output_stream)
    jd_phrases = extract_keywords_keybert(
        jd_text,
        num_keywords=25,
        keyphrase_ngram_range=(1, 4),
        diversity=0.6
    )
    print(f"Job Description Keywords ({len(jd_phrases)}): {jd_phrases}", file=output_stream)

    print("\n--- Extracting Keywords from Resume ---", file=output_stream)
    resume_phrases = extract_keywords_keybert(
        resume_text,
        num_keywords=35,
        keyphrase_ngram_range=(1, 4),
        diversity=0.7
    )
    print(f"Resume Keywords ({len(resume_phrases)}): {resume_phrases}", file=output_stream)

    jd_phrases_set = set(jd_phrases)
    resume_phrases_set = set(resume_phrases)

    matched_exact_phrases = jd_phrases_set.intersection(resume_phrases_set)

    jd_meaningful_tokens = _get_meaningful_tokens(jd_phrases)
    resume_meaningful_tokens = _get_meaningful_tokens(resume_phrases)

    common_tokens = jd_meaningful_tokens.intersection(resume_meaningful_tokens)
    total_jd_tokens = len(jd_meaningful_tokens)

    ats_score = 0
    if total_jd_tokens > 0:
        ats_score = (len(common_tokens) / total_jd_tokens) * 100

    missing_phrases_for_report = []
    missing_core_tokens_for_report = set()

    for jd_phrase in jd_phrases_set:
        if jd_phrase not in matched_exact_phrases:
            jd_phrase_tokens = _get_meaningful_tokens([jd_phrase])
            if jd_phrase_tokens: # Ensure the phrase itself has meaningful tokens
                # If the intersection is empty, it means none of the phrase's core tokens are in resume
                if not jd_phrase_tokens.intersection(resume_meaningful_tokens):
                    missing_phrases_for_report.append(jd_phrase)
                    missing_core_tokens_for_report.update(jd_phrase_tokens.difference(resume_meaningful_tokens))
            else: # If a phrase yielded no meaningful tokens (e.g., "the end"), still consider its core terms missing if relevant
                 missing_phrases_for_report.append(jd_phrase) # Could be a garbage phrase, still report as not found
                 missing_core_tokens_for_report.update(jd_phrase_tokens.difference(resume_meaningful_tokens))


    # If all phrases were matched, but still some individual tokens are missing (e.g. from longer resume phrases)
    # this will capture the remaining individual missing tokens from JD.
    if ats_score < 100:
        missing_core_tokens_for_report.update(jd_meaningful_tokens.difference(resume_meaningful_tokens))


    # --- Print Report ---
    print("\n" + "="*50, file=output_stream)
    print("          ATS Matching Report          ", file=output_stream)
    print("="*50, file=output_stream)
    print(f"Improved ATS Score (based on core terms): {ats_score:.2f}%", file=output_stream)
    print("\n--- Exactly Matched Keyphrases (JD phrases found verbatim in Resume) ---", file=output_stream)
    if matched_exact_phrases:
        for phrase in sorted(list(matched_exact_phrases)):
            print(f"- {phrase}", file=output_stream)
    else:
        print("No exact phrase matches found based on extraction.", file=output_stream)

    print("\n--- Missing JD Keyphrases (not exactly matched AND their core terms are largely absent) ---", file=output_stream)
    if missing_phrases_for_report:
        for phrase in sorted(list(missing_phrases_for_report)):
            print(f"- {phrase}", file=output_stream)
    else:
        print("All job description keyphrases were either exactly matched or their core terms were found.", file=output_stream)

    if missing_core_tokens_for_report:
        print("\n--- Key Core Terms from JD Missing in Resume (Individual words from JD that are not present in resume's keywords) ---", file=output_stream)
        for token in sorted(list(missing_core_tokens_for_report)):
            print(f"- {token}", file=output_stream)
    else:
        print("All core terms from the job description's keywords were found in your resume's keywords.", file=output_stream)


    print("\n--- All Extracted Job Description Keyphrases (for reference) ---", file=output_stream)
    for keyword in sorted(list(jd_phrases_set)):
        print(f"- {keyword}", file=output_stream)

    print("\n--- All Extracted Resume Keyphrases (for reference) ---", file=output_stream)
    for keyword in sorted(list(resume_phrases_set)):
        print(f"- {keyword}", file=output_stream)

    print("\n" + "="*50, file=output_stream)
    print("Report End", file=output_stream)
    print("="*50, file=output_stream)

    return {
        "ats_score": ats_score,
        "matched_phrases": sorted(list(matched_exact_phrases)),
        "missing_phrases": sorted(list(set(missing_phrases_for_report))),
        "missing_core_tokens": sorted(list(missing_core_tokens_for_report)),
        "jd_extracted_keywords": sorted(list(jd_phrases_set)),
        "resume_extracted_keywords": sorted(list(resume_phrases_set))
    }

# --- Main script execution ---
if __name__ == "__main__":
    print("Welcome to the ATS Score Generator!")
    print("This tool extracts keywords from a Job Description and your Resume,")
    print("then calculates a simplified ATS score and identifies missing keywords.")

    jd_file_path = input("\nEnter the file path for the Job Description (e.g., job_description.pdf): ").strip()
    resume_file_path = input("Enter the file path for your Resume (e.g., my_resume.docx): ").strip()

    job_description_content = extract_text_from_file(jd_file_path)
    resume_content = extract_text_from_file(resume_file_path)

    if not job_description_content:
        print(f"Could not read content from Job Description file: {jd_file_path}. Please check the path and file type.")
    elif not resume_content:
        print(f"Could not read content from Resume file: {resume_file_path}. Please check the path and file type.")
    else:
        # 1. Prepare for saving the report
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True) # Create 'reports' folder if it doesn't exist

        # Extract base names for filename
        resume_base_name = os.path.splitext(os.path.basename(resume_file_path))[0]
        jd_base_name = os.path.splitext(os.path.basename(jd_file_path))[0]

        # Sanitize names for filename (replace non-alphanumeric with underscore)
        resume_base_name = re.sub(r'[^\w\s-]', '_', resume_base_name).strip().replace(' ', '_')
        jd_base_name = re.sub(r'[^\w\s-]', '_', jd_base_name).strip().replace(' ', '_')

        # Find the next available number 'n'
        base_filename_prefix = f"{resume_base_name}_{jd_base_name}"
        n = 1
        while True:
            report_filename = os.path.join(reports_dir, f"{base_filename_prefix}_{n}.txt")
            if not os.path.exists(report_filename):
                break
            n += 1

        # 2. Capture the report output
        old_stdout = sys.stdout # Store original stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output # Redirect stdout to our StringIO object

        try:
            # 3. Call the function, which will now print to redirected_output
            results = calculate_ats_score_and_report(resume_content, job_description_content, output_stream=sys.stdout)
        finally:
            sys.stdout = old_stdout # Restore original stdout regardless of errors

        # 4. Get the captured string and save to file
        full_report_string = redirected_output.getvalue()

        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(full_report_string)
            print(f"\nReport saved successfully to: {report_filename}")
        except IOError as e:
            print(f"\nError saving report to file {report_filename}: {e}")

        # Optionally print to console as well (if you want it both on screen and in file)
        # print(full_report_string)