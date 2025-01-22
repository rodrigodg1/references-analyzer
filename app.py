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
from statistics import mean, median, stdev  # Import statistics here

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

    def get_yearly_publication_stats(self):
        """Get yearly publication statistics."""
        yearly_counts = Counter()
        for entry in self.entries:
            if 'year' in entry:
                year = int(entry['year'])
                yearly_counts[year] += 1
        return dict(sorted(yearly_counts.items())) # Sort by year

    def get_keyword_statistics(self):
        """Analyze keyword statistics."""
        keywords = []
        for entry in self.entries:
            if 'keywords' in entry:
                keywords.extend([kw.strip() for kw in entry['keywords'].replace(';', ',').split(',')]) # Split by comma and semicolon
            elif 'keyword' in entry: # Some bib files use 'keyword' instead of 'keywords'
                keywords.extend([kw.strip() for kw in entry['keyword'].replace(';', ',').split(',')])
        return Counter(keywords)

    def get_yearly_venue_stats(self):
        """Analyze venue distribution over years."""
        yearly_venue_counts = defaultdict(Counter)
        for entry in self.entries:
            if 'year' in entry:
                year = int(entry['year'])
                venue = None
                if 'journal' in entry:
                    venue = entry['journal']
                elif 'booktitle' in entry:
                    venue = entry['booktitle']
                if venue:
                    yearly_venue_counts[year][venue] += 1
        return {year: dict(venue_counter) for year, venue_counter in sorted(yearly_venue_counts.items())} # Sort by year

    def get_page_stats_by_type(self):
        """Get page statistics per publication type."""
        page_stats_by_type = defaultdict(list)
        for entry in self.entries:
            entry_type = entry['ENTRYTYPE']
            if 'pages' in entry:
                try:
                    pages = entry['pages'].split('--')
                    if len(pages) == 2:
                        count = int(pages[1]) - int(pages[0]) + 1
                        page_stats_by_type[entry_type].append(count)
                except (ValueError, IndexError):
                    continue

        results = {}
        for entry_type, page_counts in page_stats_by_type.items():
            if page_counts:
                results[entry_type] = {
                    'mean_pages': round(mean(page_counts), 2),
                    'median_pages': median(page_counts),
                    'std_pages': round(stdev(page_counts), 2) if len(page_counts) > 1 else 0,
                    'min_pages': min(page_counts),
                    'max_pages': max(page_counts)
                }
        return results

    def get_yearly_publication_chart_data(self):
        """Prepare data for yearly publication chart."""
        yearly_stats = self.get_yearly_publication_stats()
        years = list(yearly_stats.keys())
        counts = list(yearly_stats.values())
        return {
            'labels': years,
            'datasets': [{
                'label': 'Publications per Year',
                'data': counts,
                'backgroundColor': 'rgba(54, 162, 235, 0.7)',
                'borderColor': 'rgba(54, 162, 235, 1)',
                'borderWidth': 1
            }]
        }

    def get_entry_type_chart_data(self):
        """Prepare data for entry type chart."""
        entry_types = self.get_entry_types()
        labels = list(entry_types.keys())
        counts = list(entry_types.values())
        background_colors = [
            'rgba(255, 99, 132, 0.7)', 'rgba(255, 159, 64, 0.7)', 'rgba(255, 205, 86, 0.7)',
            'rgba(75, 192, 192, 0.7)', 'rgba(54, 162, 235, 0.7)', 'rgba(153, 102, 255, 0.7)',
            'rgba(201, 203, 207, 0.7)'
        ] # More colors if needed
        border_colors = [color.replace('0.7', '1') for color in background_colors]

        return {
            'labels': labels,
            'datasets': [{
                'label': 'Publication Types',
                'data': counts,
                'backgroundColor': background_colors[:len(labels)], # Use only needed colors
                'borderColor': border_colors[:len(labels)],
                'borderWidth': 1
            }]
        }

    def get_top_venues_chart_data(self, top_n=10):
        """Prepare data for top venues chart."""
        top_venues = self.get_venue_statistics().most_common(top_n)
        venues = [venue for venue, count in top_venues]
        counts = [count for venue, count in top_venues]
        return {
            'labels': venues,
            'datasets': [{
                'label': 'Publications',
                'data': counts,
                'backgroundColor': 'rgba(255, 205, 86, 0.7)',
                'borderColor': 'rgba(255, 205, 86, 1)',
                'borderWidth': 1
            }]
        }

    def get_top_authors_chart_data(self, top_n=10):
        """Prepare data for top authors chart."""
        top_authors = self.get_author_statistics()[0].most_common(top_n)
        authors = [author for author, count in top_authors]
        counts = [count for author, count in top_authors]
        return {
            'labels': authors,
            'datasets': [{
                'label': 'Publications',
                'data': counts,
                'backgroundColor': 'rgba(75, 192, 192, 0.7)',
                'borderColor': 'rgba(75, 192, 192, 1)',
                'borderWidth': 1
            }]
        }

    def get_keyword_chart_data(self, top_n=15):
        """Prepare data for keyword chart."""
        top_keywords = self.get_keyword_statistics().most_common(top_n)
        keywords = [keyword for keyword, count in top_keywords]
        counts = [count for keyword, count in top_keywords]
        return {
            'labels': keywords,
            'datasets': [{
                'label': 'Keyword Frequency',
                'data': counts,
                'backgroundColor': 'rgba(153, 102, 255, 0.7)',
                'borderColor': 'rgba(153, 102, 255, 1)',
                'borderWidth': 1
            }]
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
            'citation_stats': analyzer.validate_citations(),
            'yearly_stats': analyzer.get_yearly_publication_stats(),
            'keyword_stats': analyzer.get_keyword_statistics(),
            'yearly_venue_stats': analyzer.get_yearly_venue_stats(),
            'page_stats_by_type': analyzer.get_page_stats_by_type(),

            # Chart data:
            'yearly_publication_chart': analyzer.get_yearly_publication_chart_data(),
            'entry_type_chart': analyzer.get_entry_type_chart_data(),
            'top_venues_chart': analyzer.get_top_venues_chart_data(),
            'top_authors_chart': analyzer.get_top_authors_chart_data(),
            'keyword_chart': analyzer.get_keyword_chart_data()
        }

        return render_template('results.html', results=results)

    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return f'Error analyzing files: {str(e)}', 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)