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
from statistics import mean, median, stdev  
from urllib.parse import quote  # Add this with other imports
import requests
from bibtexparser.bparser import BibTexParser
from bibtexparser.bwriter import BibTexWriter
import networkx as nx

from itertools import combinations

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
        self.label_reference_stats = self._analyze_labels_and_references() if tex_content else None # Analyze labels and references

    

    def get_collaboration_network(self):
        """Analyze co-author relationships and collaboration patterns."""
        G = nx.Graph()
        author_publications = defaultdict(int)
        collaboration_strength = defaultdict(int)

        for entry in self.entries:
            if 'author' in entry:
                authors = [a.strip() for a in entry['author'].split(' and ')]
                # Update individual author counts
                for author in authors:
                    author_publications[author] += 1
                # Update collaboration strengths
                for pair in combinations(sorted(authors), 2):
                    collaboration_strength[pair] += 1
                G.add_nodes_from(authors)
                G.add_edges_from(combinations(authors, 2))

        # Calculate network metrics
        try:
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            communities = nx.algorithms.community.greedy_modularity_communities(G)
        except Exception as e:
            app.logger.error(f"Network analysis error: {str(e)}")
            centrality = {}
            betweenness = {}
            communities = []

        return {
            'total_authors': len(author_publications),
            'avg_authors_per_paper': self.get_basic_stats()['avg_authors_per_paper'],
            'most_prolific_author': max(author_publications.items(), key=lambda x: x[1], default=('None', 0)),
            'strongest_collaboration': max(collaboration_strength.items(), key=lambda x: x[1], default=(('None', 'None'), 0)),
            'centrality': centrality,
            'betweenness': betweenness,
            'communities': communities,
            'network_data': nx.node_link_data(G)
        }

    def get_temporal_analysis(self):
        """Analyze publication patterns over time."""
        years = defaultdict(lambda: {
            'count': 0,
            'authors': set(),
            'keywords': Counter(),
            'venues': Counter()
        })

        for entry in self.entries:
            if 'year' in entry:
                year = entry['year']
                years[year]['count'] += 1
                
                if 'author' in entry:
                    authors = [a.strip() for a in entry['author'].split(' and ')]
                    years[year]['authors'].update(authors)
                
                if 'keywords' in entry:
                    keywords = [kw.strip() for kw in entry['keywords'].replace(';', ',').split(',')]
                    years[year]['keywords'].update(keywords)
                
                if 'journal' in entry:
                    years[year]['venues'][entry['journal']] += 1
                elif 'booktitle' in entry:
                    years[year]['venues'][entry['booktitle']] += 1

        # Calculate trends
        trend_metrics = {
            'year': [],
            'publications': [],
            'new_authors': [],
            'keyword_trends': defaultdict(list),
            'venue_trends': defaultdict(list)
        }

        prev_authors = set()
        for year in sorted(years.keys()):
            trend_metrics['year'].append(year)
            trend_metrics['publications'].append(years[year]['count'])
            
            # New authors calculation
            current_authors = years[year]['authors']
            new_authors = len(current_authors - prev_authors)
            trend_metrics['new_authors'].append(new_authors)
            prev_authors.update(current_authors)
            
            # Keyword trends
            for kw, count in years[year]['keywords'].most_common(5):
                trend_metrics['keyword_trends'][kw].append((year, count))
                
            # Venue trends
            for venue, count in years[year]['venues'].most_common(3):
                trend_metrics['venue_trends'][venue].append((year, count))

        return trend_metrics

    def get_content_analysis(self):
        """Analyze paper content through titles and abstracts."""
        title_words = Counter()
        abstract_words = Counter()
        bigrams = Counter()
        trigrams = Counter()

        for entry in self.entries:
            # Title analysis
            if 'title' in entry:
                title = entry['title'].lower()
                words = re.findall(r'\w{4,}', title)
                title_words.update(words)
                
                # Generate n-grams
                bigrams.update(zip(words, words[1:]))
                trigrams.update(zip(words, words[1:], words[2:]))
            
            # Abstract analysis
            if 'abstract' in entry:
                abstract = entry['abstract'].lower()
                abstract_words.update(re.findall(r'\w{4,}', abstract))

        # Calculate keyness scores
        total_title = sum(title_words.values())
        total_abstract = sum(abstract_words.values())
        keyness = {}
        for word in set(title_words) | set(abstract_words):
            title_freq = title_words[word]/total_title if total_title else 0
            abstract_freq = abstract_words[word]/total_abstract if total_abstract else 0
            keyness[word] = title_freq - abstract_freq

        return {
            'title_wordcloud': title_words.most_common(50),
            'abstract_wordcloud': abstract_words.most_common(50),
            'key_terms': sorted(keyness.items(), key=lambda x: x[1], reverse=True)[:50],
            'bigrams': bigrams.most_common(20),
            'trigrams': trigrams.most_common(20)
        }

    def get_institutional_analysis(self):
        """Analyze author affiliations and institutions."""
        institutions = Counter()
        countries = Counter()
        domains = Counter()

        for entry in self.entries:
            if 'author' in entry and 'affiliation' in entry:
                affils = entry['affiliation'].split(';')
                for affil in affils:
                    affil = affil.strip()
                    if not affil:
                        continue
                    
                    # Extract domain from email
                    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b', affil)
                    if email_match:
                        domain = email_match.group(1).lower()
                        domains[domain] += 1
                    
                    # Extract country
                    country_match = re.search(r'\b(?:USA|United States|China|Germany|UK|United Kingdom|France|Japan|India)\b', affil, re.I)
                    if country_match:
                        countries[country_match.group(0).title()] += 1
                    
                    # Institution name
                    institution = textwrap.shorten(affil, width=60, placeholder="...")
                    institutions[institution] += 1

        return {
            'top_institutions': institutions.most_common(20),
            'country_distribution': countries,
            'email_domains': domains.most_common(20)
        }

    def get_citation_analysis(self):
        """Deep analysis of citation patterns and impact."""
        cited_years = []
        citation_ages = []
        citation_velocity = defaultdict(list)
        highly_cited = []

        current_year = datetime.now().year
        for entry in self.entries:
            if 'year' in entry:
                try:
                    pub_year = int(entry['year'])
                    cited_years.append(pub_year)
                    
                    # Citation age analysis
                    if self.citations and entry['ID'] in self.citations:
                        age = current_year - pub_year
                        citation_ages.append(age)
                        citation_velocity[pub_year].append(self.citations[entry['ID']])
                except ValueError:
                    continue

        # Calculate citation impact metrics
        citation_velocity = {year: sum(counts)/len(counts) for year, counts in citation_velocity.items()}
        avg_citation_age = sum(citation_ages)/len(citation_ages) if citation_ages else 0

        # Identify highly cited papers (top 5%)
        if self.citations:
            citation_counts = list(self.citations.values())
            threshold = sorted(citation_counts, reverse=True)[len(citation_counts)//20]
            highly_cited = [entry for entry in self.entries 
                           if entry['ID'] in self.citations 
                           and self.citations[entry['ID']] >= threshold]

        return {
            'avg_citation_age': round(avg_citation_age, 1),
            'citation_velocity': citation_velocity,
            'citation_distribution': Counter(cited_years),
            'highly_cited_papers': highly_cited[:10],
            'citation_network': self._build_citation_network()
        }




    # In the BibliometricAnalyzer class
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
                            'context': line.strip()  # Store context as string
                        })

        self.citation_locations = citation_locations
        return citations

    # Update the citation network builder
    def _build_citation_network(self):
        """Build citation network graph if LaTeX references are available."""
        if not self.tex_content:
            return None
        
        G = nx.DiGraph()
        
        # Add all papers as nodes
        for entry in self.entries:
            G.add_node(entry['ID'], **entry)
        
        # Add citation edges
        for cite, locations in self.citation_locations.items():
            if cite in G.nodes:
                # Create edges from context lines to citations
                for loc in locations:
                    context_str = loc['context']  # Get the actual line text
                    G.add_edge(context_str, cite)  # Link context line to citation

        return {
            'nodes': list(G.nodes),
            'edges': list(G.edges),
            'most_cited': sorted(G.in_degree, key=lambda x: x[1], reverse=True)[:10]
        }






















    def _analyze_labels_and_references(self):
        """Analyze labels and references for figures, tables, algorithms, and equations and return detailed status."""
        element_types = {
            'figure': {'label_pattern': r'\\label{fig:(.*?)}', 'ref_pattern': r'\\ref{fig:(.*?)}'},
            'table': {'label_pattern': r'\\label{tab:(.*?)}', 'ref_pattern': r'\\ref{tab:(.*?)}'},
            'algorithm': {'label_pattern': r'\\label{alg:(.*?)}', 'ref_pattern': r'\\ref{alg:(.*?)}'},
            'equation': {'label_pattern': r'\\label{eq:(.*?)}', 'ref_pattern': r'\\ref{eq:(.*?)}'}
        }

        stats = {}
        for element_type, patterns in element_types.items():
            label_pattern = patterns['label_pattern']
            ref_pattern = patterns['ref_pattern']

            defined_labels = re.findall(label_pattern, self.tex_content)
            referenced_labels = re.findall(ref_pattern, self.tex_content)
            referenced_labels_set = set(referenced_labels) # For efficient checking

            detailed_labels_status = []
            for label in defined_labels:
                is_referenced = label in referenced_labels_set
                detailed_labels_status.append({'label': label, 'referenced': is_referenced})

            missing_references = [label_status['label'] for label_status in detailed_labels_status if not label_status['referenced']]
            unused_references = list(set(referenced_labels) - set(defined_labels)) # Unused refs remain the same logic

            stats[element_type] = {
                'detailed_label_status': detailed_labels_status, # Detailed status for each label
                'total_defined': len(defined_labels),
                'total_referenced': len(referenced_labels),
                'missing_references': missing_references, # Keep missing references for summary if needed
                'unused_references': unused_references,
            }
        return stats

    def get_label_reference_stats_data(self):
        """Get statistics about labels and references for figures, tables, algorithms, and equations."""
        return self.label_reference_stats if self.label_reference_stats else {}


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

    bib_content = None  # Initialize bib_content outside the try block
    tex_content = None  # Initialize tex_content outside the try block

    try:
        bib_content = bib_file.read().decode('utf-8')

        # Process optional tex file
        if 'tex_file' in request.files and request.files['tex_file'].filename:
            tex_file = request.files['tex_file']
            if not allowed_file(tex_file.filename):
                return 'Invalid file type for LaTeX file', 400
            tex_content = tex_file.read().decode('utf-8')

        analyzer = BibliometricAnalyzer(bib_content, tex_content) # Instantiate analyzer here

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
            'label_reference_stats': analyzer.get_label_reference_stats_data(), # Get label reference stats

            'collaboration_network': analyzer.get_collaboration_network(),
            'temporal_analysis': analyzer.get_temporal_analysis(),
            'content_analysis': analyzer.get_content_analysis(),
            'institutional_analysis': analyzer.get_institutional_analysis(),
            'citation_analysis': analyzer.get_citation_analysis(),


            # Chart data:
            'yearly_publication_chart': analyzer.get_yearly_publication_chart_data(),
            'entry_type_chart': analyzer.get_entry_type_chart_data(),
            'top_venues_chart': analyzer.get_top_venues_chart_data(),
            'top_authors_chart': analyzer.get_top_authors_chart_data(),
            'keyword_chart': analyzer.get_keyword_chart_data()


        }

        return render_template('results.html', results=results)

    except UnicodeDecodeError:
        return 'Invalid BibTeX or LaTeX file encoding (must be UTF-8)', 400 # Catch encoding errors here
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return f'Error analyzing files: {str(e)}', 500




# Add to your existing routes
@app.route('/doi2bib')
def doi2bib():
    """Render the DOI to BibTeX conversion page."""
    return render_template('doi2bib.html')





@app.route('/convert-dois', methods=['POST'])
@limiter.limit("10 per minute")
def convert_dois():
    """Convert DOIs from text input to BibTeX entries with DOI keys."""
    dois = request.form.get('dois', '')
    
    if not dois:
        return 'No DOIs entered', 400

    try:
        clean_dois = [doi.strip() for doi in dois.splitlines() if doi.strip()]
        if not clean_dois:
            return 'No valid DOIs entered', 400

        bib_entries = []
        headers = {'User-Agent': 'doi2bib-flask/1.0 (mailto:your@email.com)'}
        writer = BibTexWriter()
        writer.indent = '  '
        writer.order_entries_by = ('author', 'title', 'journal', 'year', 
                                  'volume', 'number', 'pages', 'doi', 'url')

        successful_dois = []
        failed_dois = []

        for doi in clean_dois:
            try:
                doi = doi.replace('"', '').replace("'", "").split()[0]  # Handle any extra text
                encoded_doi = quote(doi, safe='')
                url = f"https://api.crossref.org/works/{encoded_doi}/transform/application/x-bibtex"
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Create new parser instance for each DOI
                parser = BibTexParser(common_strings=True)
                parser.ignore_nonstandard_types = False
                bib_db = bibtexparser.loads(response.text, parser=parser)
                
                # Handle multiple entries by taking the first valid one
                if not bib_db.entries:
                    raise ValueError("No entries found in API response")
                
                # Select first entry that looks like a paper
                valid_entry = next((e for e in bib_db.entries 
                                  if e.get('ENTRYTYPE') in ['article', 'inproceedings', 'phdthesis']), None)
                
                if not valid_entry:
                    raise ValueError("No valid entry type in response")
                
                # Force DOI-based citation key and fields
                valid_entry['ID'] = doi
                valid_entry['doi'] = doi
                
                # Create new database with just this entry
                new_db = bibtexparser.bibdatabase.BibDatabase()
                new_db.entries = [valid_entry]
                
                formatted_entry = writer.write(new_db).strip()
                bib_entries.append(formatted_entry)
                successful_dois.append(doi)

            except Exception as e:
                failed_dois.append(doi)
                app.logger.warning(f"Failed {doi}: {str(e)}")
                continue

        if not bib_entries:
            return 'No valid BibTeX entries could be retrieved', 400

        output = '\n\n'.join(bib_entries)
        app.logger.info(f"Converted {len(successful_dois)}/{len(clean_dois)} DOIs")
        
        mem_file = io.BytesIO(output.encode('utf-8'))
        mem_file.seek(0)
        
        response = send_file(
            mem_file,
            as_attachment=True,
            download_name="references.bib",
            mimetype="text/x-bibtex"
        )
        
        response.headers['X-Success-Count'] = str(len(successful_dois))
        response.headers['X-Failed-DOIs'] = ','.join(failed_dois)
        
        return response

    except Exception as e:
        app.logger.error(f"System error: {str(e)}")
        return f'Conversion failed: {str(e)}', 500







if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)