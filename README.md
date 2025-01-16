# Bibliometric Analysis Application

A web application for analyzing BibTeX files and LaTeX citations. This application processes bibliographic data and generates statistical analysis of publications, authors, and citations.

## Prerequisites

- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git (optional)

## Installation Steps

1. Install Docker:
   - Windows/Mac: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Linux: Follow [Docker Engine installation](https://docs.docker.com/engine/install/)

2. Verify Docker installation:
```bash
docker --version
docker-compose --version
```

3. Clone or download the project:
```bash
git clone [repository-url]
# or download and extract the ZIP file
```

4. Navigate to project directory:
```bash
cd biblio-analyzer
```

## Running the Application

1. Start Docker Desktop (Windows/Mac) or Docker service (Linux):
```bash
# Linux only
sudo systemctl start docker
```

2. Build and start the application:
```bash
docker-compose up --build
```

3. Access the application:
- Local machine: `http://localhost:5000`
- Network access: `http://[server-ip]:5000`

4. Stop the application:
```bash
docker-compose down
```

## Troubleshooting

### Docker Desktop Not Running
Windows/Mac:
1. Open Docker Desktop
2. Wait for engine to start
3. Check status in system tray

Linux:
```bash
sudo systemctl status docker
```

### Port Already in Use
```bash
# Check port usage
netstat -ano | findstr :5000    # Windows
netstat -tulpn | grep :5000     # Linux/Mac

# Modify port in docker-compose.yml if needed:
ports:
  - "8080:5000"    # Changes external port to 8080
```

### Permission Issues
```bash
# Linux/Mac
sudo chmod 755 uploads
sudo chown -R $USER:$USER uploads
```

### Container Logs
```bash
docker-compose logs -f
```

## Usage Guidelines

1. File Upload:
   - BibTeX files (.bib extension)
   - LaTeX files (.tex extension, optional)
   - Maximum file size: 16MB

2. Analysis Features:
   - Publication statistics
   - Author analysis
   - Citation validation
   - Page statistics

## Project Structure
```
biblio-analyzer/
├── app.py               # Main application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Service configuration
├── templates/          # HTML templates
└── uploads/           # File storage
```

## System Requirements

- RAM: 2GB minimum
- Storage: 1GB free space
- Network: Open port 5000
- Operating System:
  - Windows 10/11 Pro/Enterprise/Education
  - macOS 10.15 or newer
  - Linux with kernel 4.0 or newer