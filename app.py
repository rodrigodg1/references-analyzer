# app.py
from flask import Flask, render_template, request, send_file, session, abort
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import bibtexparser
from collections import Counter, defaultdict
import os
import re
from datetime import datetime
import io
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Application Configuration
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', os.urandom(24)),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER='uploads',
    ALLOWED_EXTENSIONS={'bib', 'tex'}
)

# Rate limiting configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class BibliometricAnalyzer:
    def __init__(self, bibtex_content, tex_content=None):
        """Initialize analyzer with BibTeX content and optional LaTeX content."""
        self.bib_database = bibtexparser.loads(bibtex_content)
        self.entries = self.bib_database.entries
        self.tex_content = tex_content
        self.citations = self._extract_citations() if tex_content else None

    def _extract_citations(self):
        """Extract citations from LaTeX content."""
        cite_patterns = [
            r'\\cite{(.*?)}',
            r'\\citep{(.*?)}',
            r'\\citet{(.*?)}',
            r'\\citeyear{(.*?)}',
            r'\\citeauthor{(.*?)}'
        ]

        citations = Counter()
        citation_locations = defaultdict(list)
        lines = self.tex_content.split('\n')

        for line_num, line in enumerate(lines, 1):
            for pattern in cite_patterns:
                for match in re.finditer(pattern, line):
                    keys = match.group(1).split(',')
                    for key in keys:
                        key = key.strip()
                        citations[key] += 1
                        citation_locations[key].append({
                            'line': line_num,
                            'context': line.strip()
                        })

        self.citation_locations = citation_locations
        return citations

    def validate_citations(self):
        """Validate citations between BibTeX and LaTeX files."""
        if not self.citations:
            return None
            
        bib_keys = {entry['ID'] for entry in self.entries}
        tex_citations = set(self.citations.keys())

        return {
            'missing_citations': {
                cite: self.citation_locations[cite] 
                for cite in tex_citations - bib_keys
            },
            'unused_citations': list(bib_keys - tex_citations),
            'valid_citations': len(tex_citations & bib_keys),
            'total_citations': len(tex_citations),
            'total_references': len(bib_keys)
        }

    def get_venue_statistics(self):
        """Analyze publication venues."""
        venues = []
        for entry in self.entries:
            if 'journal' in entry:
                venues.append(entry['journal'])
            elif 'booktitle' in entry:
                venues.append(entry['booktitle'])
        return Counter(venues)

    def get_author_statistics(self):
        """Get author statistics."""
        author_counts = Counter()
        author_years = defaultdict(list)

        for entry in self.entries:
            if 'author' in entry and 'year' in entry:
                year = int(entry['year'])
                authors = [author.strip() for author in entry['author'].split(' and ')]
                for author in authors:
                    author_counts[author] += 1
                    author_years[author].append(year)

        return author_counts, author_years

    def get_entry_types(self):
        """Get publication type statistics."""
        return Counter(entry['ENTRYTYPE'] for entry in self.entries)

    def get_page_statistics(self):
        """Get page number statistics."""
        page_counts = []
        for entry in self.entries:
            if 'pages' in entry:
                try:
                    pages = entry['pages'].split('--')
                    if len(pages) == 2:
                        count = int(pages[1]) - int(pages[0]) + 1
                        page_counts.append(count)
                except (ValueError, IndexError):
                    continue

        if page_counts:
            from statistics import mean, median, stdev
            return {
                'mean_pages': round(mean(page_counts), 2),
                'median_pages': median(page_counts),
                'std_pages': round(stdev(page_counts), 2) if len(page_counts) > 1 else 0,
                'min_pages': min(page_counts),
                'max_pages': max(page_counts)
            }
        return {}

    def get_basic_stats(self):
        """Calculate basic statistics."""
        years = [int(entry['year']) for entry in self.entries if 'year' in entry]
        authors = []
        for entry in self.entries:
            if 'author' in entry:
                authors.extend([a.strip() for a in entry['author'].split(' and ')])

        return {
            'total_entries': len(self.entries),
            'year_range': f"{min(years)} - {max(years)}" if years else "N/A",
            'total_venues': len(set(self.get_venue_statistics())),
            'total_authors': len(set(authors)),
            'avg_authors_per_paper': round(len(authors) / len(self.entries), 2)
        }

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded error."""
    return 'File too large (max 16MB)', 413

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def analyze():
    """Process and analyze uploaded files."""
    # Validate bib file
    if 'bib_file' not in request.files:
        return 'No BibTeX file uploaded', 400
    
    bib_file = request.files['bib_file']
    if bib_file.filename == '':
        return 'No BibTeX file selected', 400

    if not allowed_file(bib_file.filename):
        return 'Invalid file type for BibTeX file', 400

    try:
        bib_content = bib_file.read().decode('utf-8')
    except UnicodeDecodeError:
        return 'Invalid BibTeX file encoding (must be UTF-8)', 400

    # Process optional tex file
    tex_content = None
    if 'tex_file' in request.files and request.files['tex_file'].filename:
        tex_file = request.files['tex_file']
        if not allowed_file(tex_file.filename):
            return 'Invalid file type for LaTeX file', 400
        
        try:
            tex_content = tex_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return 'Invalid LaTeX file encoding (must be UTF-8)', 400

    try:
        analyzer = BibliometricAnalyzer(bib_content, tex_content)
        
        results = {
            'basic_stats': analyzer.get_basic_stats(),
            'entry_types': analyzer.get_entry_types(),
            'venues': analyzer.get_venue_statistics(),
            'author_stats': analyzer.get_author_statistics()[0],
            'page_stats': analyzer.get_page_statistics(),
            'citation_stats': analyzer.validate_citations()
        }
        
        return render_template('results.html', results=results)
    
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return f'Error analyzing files: {str(e)}', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)