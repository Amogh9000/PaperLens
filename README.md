# Automated OMR Evaluation & Scoring System

**PaperLens - Advanced OMR Processing Platform**

A sophisticated Streamlit-based frontend for automated OMR (Optical Mark Recognition) sheet evaluation and scoring. This application provides a professional, user-friendly interface for educators and evaluators to process OMR sheets efficiently.

## Features

### Core Functionality
- **File Upload**: Browse or drag-and-drop PNG, JPG, JPEG, and PDF files
- **Real-time Processing**: On-device OMR evaluation with step-wise progress
- **Comprehensive Results**: Subject-wise scoring with detailed breakdowns
- **Annotated Visualization**: Review view with detected bubble overlays
- **Export Options**: Download results as CSV and Excel
- **Dashboard Analytics**: Aggregate statistics and performance insights (demo data)

### User Experience
- **Clean, Professional UI**: Intuitive layout optimized for evaluators
- **Responsive Layout**: Works across common desktop resolutions
- **Interactive Charts**: Altair-based visual summaries
- **Help & Documentation**: In-app hints and this README
- **Progress Indicators**: Real-time feedback during processing

## Quick Start

### Prerequisites
- Python 3.10–3.12
- pip (comes with Python)

### Installation

1. **Clone or download the project**
   ```powershell
   cd D:\OMR
   ```

2. **Create and activate a virtual environment (Windows PowerShell)**
   ```powershell
   # from project root
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```powershell
   # use python -m to avoid PATH issues
   python -m streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:8501`
   - The application auto-opens in your default browser

### One-liner setup (optional)

If you keep a setup script, you can automate environment creation/activation and installs. Otherwise, follow the steps above.

## Usage Guide

### 1. Upload OMR Sheet
- Go to the `Upload` tab
- Click to browse or drag-and-drop a file
- Supported formats: PNG, JPG, JPEG, PDF (recommended scan ≥ 300 DPI)

### 2. Select Answer Key Set
- Use the `Answer Key Set` dropdown at the top: choose `A` or `B`
- Mapping in `answer_key.json`:
  - Set A → `set_1`
  - Set B → `set_2`

### 3. Start Evaluation
- Click `Run Evaluation`
- Watch the progress bar and status messages

### 4. Review Results
- **Overall Score**: Total score (and accuracy) from the engine
- **Subject Breakdown**: Scores for the 5 subjects (20 Qs each)
- **Annotated Image**: Visual feedback of detected bubbles
- **Detailed Table**: Per-question results

### 5. Download Results
- **CSV** and **Excel**: Export detailed results

##  Architecture

### Frontend
- **Streamlit** (`app.py`) orchestrates the UI (Upload, Results, Review, Dashboard)
- **Altair** for charts; **Pandas** drives tabular displays and exports

### Engine (Backend Processing)
- **OpenCV + NumPy** (`engine.py`) for perspective correction, thresholding, and bubble detection
- **PyMuPDF (fitz)**: Renders first page of PDFs into images for processing
- **Answer Keys**: Loaded from `answer_key.json` (top-level `set_1` and `set_2`) or Excel

### Data Flow (High level)
1. Upload image/PDF → image decoded (or rendered)
2. `engine.flatten_image(path)` to deskew/flatten
3. Detect answer area and bubbles; group bubbles into questions
4. Grade against selected set’s key
5. Return results + review visuals to the UI

## Project Structure

```
OMR/
├── app.py               # Main Streamlit application (UI and orchestration)
├── engine.py            # OMR processing engine (OpenCV)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation (this file)
├── answer_key.json      # Default/local answer keys (set_1, set_2)
└── maps/                # Optional mapping files / examples
```

## Design Philosophy

### Professional & Clean
- Minimal UI focused on evaluator workflow
- Consistent styling and clear information hierarchy

### User-Centric
- Simple upload-and-run flow with clear messaging
- Helpful captions indicating which key/source is in use

### Scalable Architecture
- Modular engine separated from UI
- Ready for batch processing and API exposure

## Technical Specifications

### Performance
- **Processing Time**: Typically a few seconds per sheet (machine-dependent)
- **File Size Limit**: Streamlit default (recommend < 10MB)
- **Supported Formats**: PNG, JPG, JPEG, PDF

### Dependencies
- **Streamlit**: Web UI
- **Pandas/NumPy**: Data processing
- **Pillow/OpenPyXL**: Image and Excel handling
- **OpenCV**: Vision engine
- **PyMuPDF**: PDF rendering

## Future Enhancements

### Planned Features
- **Batch Processing** of multiple sheets
- **Richer Analytics** with more charts and filters
- **PDF Reports** summarizing results
- **API Integration** for external systems

### Backend Extensions
- Advanced bubble classifiers and adaptive heuristics
- Optional database persistence for large-scale runs

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Docstrings for functions/classes
- Type hints where useful
- Add basic tests for engine helpers

## Support & Contact

**PaperLens Support Team**
- Email: support@paperlens.ai
- Website: www.paperlens.ai
- Support Hours: 9 AM – 6 PM (Mon–Fri)

## Engine CLI Usage (optional)

Run the engine locally from the command line (example):

```bash
python engine.py --input path/to/omr_image.jpg \
                 --key answer_key.json \
                 --out result.json \
                 --overlay overlay.png
```

Outputs:
- `result.json` with: `student_id`, `subject_scores`, `total_score`, `evaluation_time`, `accuracy`
- `overlay.png` annotated sheet preview

## Backend API (FastAPI) — Optional

If you plan to expose an API:

```bash
# Install optional deps
pip install fastapi uvicorn python-multipart

# Start the server
uvicorn fastapi_app:app --reload
```

- Health: `GET http://localhost:8000/health`
- Evaluate: `POST http://localhost:8000/evaluate`
  - Form fields: `file` (upload), `key_set` (`A` or `B`)

## Database (SQLite) — Optional

If you persist results, use SQLite via SQLAlchemy (example only):

```python
import sqlite3
con = sqlite3.connect('omr.db')
print(con.execute('SELECT COUNT(*) FROM results').fetchone())
```

## PDF Handling

- Upload PDFs directly; first page is rendered via PyMuPDF
- Scan at 300 DPI+ for best results

## License

This project is proprietary software developed by PaperLens. All rights reserved.

---

**Built with care by the PaperLens Team**

*Revolutionizing document processing through advanced AI*
