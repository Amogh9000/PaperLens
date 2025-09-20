# Automated OMR Evaluation & Scoring System

**PaperLens - Advanced OMR Processing Platform**

A sophisticated Streamlit-based frontend for automated OMR (Optical Mark Recognition) sheet evaluation and scoring system. This application provides a professional, user-friendly interface for educators and evaluators to process OMR sheets efficiently.

## Features

### Core Functionality
- **File Upload**: Drag-and-drop interface supporting PNG, JPG, JPEG, and PDF formats
- **Real-time Processing**: AI-powered evaluation with progress tracking
- **Comprehensive Results**: Subject-wise scoring with detailed breakdowns
- **Annotated Visualization**: System-generated overlay showing correct/incorrect answers
- **Export Options**: Download results in CSV and Excel formats
- **Dashboard Analytics**: Aggregate statistics and performance insights

### User Experience
- **Clean, Professional UI**: Intuitive design optimized for evaluators
- **Responsive Layout**: Works seamlessly across different screen sizes
- **Interactive Charts**: Visual representation of performance data
- **Help & Documentation**: Comprehensive user guide and FAQ section
- **Progress Indicators**: Real-time feedback during processing

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**
   ```bash
   cd OMR
   ```

2. **Create and activate a virtual environment (Windows PowerShell)**
   ```powershell
   # from project root
   python -m venv venv
   .\venv\Scripts\Activate.ps1
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
   - The application will automatically open in your default browser

### One-liner setup (recommended)

Use the helper script to create/activate `venv`, upgrade pip, and install all dependencies:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
./setup_env.ps1
```

Then run apps with:

```powershell
# Streamlit UI
python -m streamlit run app.py

# FastAPI backend
python -m uvicorn fastapi_app:app --reload
```

## Usage Guide

### 1. Upload OMR Sheet
- Navigate to the "Upload & Evaluate" section
- Drag and drop your OMR sheet or click to browse
- Supported formats: PNG, JPG, JPEG, PDF (max 10MB)

### 2. Start Evaluation
- Click "Start Evaluation" button
- Monitor progress through the real-time status updates
- Processing typically takes 2-5 seconds

### 3. Review Results
- **Overall Score**: Prominently displayed total score out of 100
- **Subject Breakdown**: Individual scores for all 5 subjects (20 questions each)
- **Annotated Image**: Visual feedback showing correct/incorrect answers
- **Detailed Table**: Comprehensive breakdown with grades and percentages

### 4. Download Results
- **CSV Format**: Structured data for further analysis
- **Excel Format**: Formatted spreadsheet with all details
- **PDF Report**: Comprehensive evaluation report (coming soon)

### 5. Dashboard Analytics
- **Performance Metrics**: Total students, average scores, pass rates
- **Visual Charts**: Subject-wise averages and grade distributions
- **Recent Activity**: Latest evaluation history

## 🏗️ Architecture

### Current Implementation (MVP)
- **Frontend**: Streamlit-based web application
- **Data**: Dummy JSON structure matching backend specifications
- **Visualization**: Altair charts and custom CSS styling
- **Export**: Pandas-based CSV/Excel generation

### Backend Integration Ready
The application is designed to seamlessly integrate with the backend `engine.py` when ready:

```python
# Current dummy data structure
DUMMY_RESULTS = {
    "student_id": "STU123",
    "subject_scores": {
        "Mathematics": 18,
        "English": 15,
        "Science": 20,
        "Logical Reasoning": 17,
        "General Knowledge": 16
    },
    "total_score": 86,
    "annotated_image": "dummy_overlay.png",
    "evaluation_time": "2024-01-15 14:30:25",
    "confidence_score": 94.5
}
```

## Project Structure

```
OMR/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Project documentation
└── engine.py          # Backend processing (to be integrated)
```

## Design Philosophy

### Professional & Clean
- Minimalist design focused on functionality
- Consistent color scheme and typography
- Logical information hierarchy

### User-Centric
- Intuitive navigation with clear sections
- Helpful tooltips and guidance
- Error handling and user feedback

### Scalable Architecture
- Modular code structure
- Easy backend integration
- Extensible for future features

## Technical Specifications

### Performance
- **Processing Time**: 2-5 seconds per sheet
- **File Size Limit**: 10MB maximum
- **Supported Formats**: PNG, JPG, JPEG, PDF
- **Accuracy**: 95%+ bubble detection (when backend integrated)

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Pillow**: Image processing
- **Altair**: Statistical visualization
- **OpenPyXL**: Excel file handling

## Future Enhancements

### Planned Features
- **Batch Processing**: Multiple sheet evaluation
- **Custom Answer Keys**: Configurable evaluation templates
- **Advanced Analytics**: Detailed performance insights
- **PDF Reports**: Comprehensive evaluation documents
- **User Management**: Multi-user support with authentication
- **API Integration**: RESTful API for external systems

### Backend Integration
- Replace dummy data with live `engine.py` calls
- Real image processing and bubble detection
- Database integration for persistent storage
- Advanced AI model integration

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings for all functions
- Include type hints where appropriate
- Write comprehensive tests

## Support & Contact

**PaperLens Support Team**
- Email: support@paperlens.ai
- Website: www.paperlens.ai
- Phone: +1-XXX-XXX-XXXX
- Support Hours: 9 AM - 6 PM (Mon-Fri)

## Build Plan & Milestones

### Phase 1: Foundations (You Are Here)
- Vision Lead (engine.py)
  - Implement `flatten_image()` using contour + perspective warp.
  - Stub bubble reader and scoring against JSON key.
  - Provide CLI to test locally.
- UI Lead (app.py)
  - Streamlit UI with tabs: Upload, Results, Review, Dashboard.
  - Dummy data support and Demo Mode.
- Data Lead
  - Deliver `maps/default_key.json` and collect test images.

Outcome: a working vision engine (CLI) and a beautiful UI (fake data ready).

### Phase 2: Integration
- Integration Lead
  - Import `engine.evaluate_omr()` in `app.py` and replace dummy data when Demo Mode is off.
- UI Lead
  - Polish UI to support real data (loading, error states).
- Data Lead
  - Act as Lead Tester; upload varied test sheets and report issues.

### Phase 3: Polish & Wow Factor
- Engine returns annotated overlay (red/green marks) as PNG bytes.
- UI displays overlay with zoom and legend, improved UX.
- Data Lead continues testing and documenting results.

### Phase 4: Finalization & Pitch Prep
- Code Freeze; no new features.
- Data Lead prepares slides; Integration Lead assists with technical pitch.

## Engine CLI Usage

Run the OMR engine locally from the command line:

```bash
python engine.py --input path/to/omr_image.jpg \
                 --key maps/default_key.json \
                 --out result.json \
                 --overlay overlay.png
```

Outputs:
- `result.json` with: `student_id`, `subject_scores`, `total_score`, `evaluation_time`, `confidence_score`.
- `overlay.png` annotated sheet preview (for demo it is simulated and deterministic).

## App Integration & Testing

- Demo Mode: Toggle on to load sample results without uploading.
- Real Evaluation: Upload a PNG/JPG/PDF and click Run Evaluation; UI will call `engine.evaluate_omr()`.
- Results Tab: Shows total score, subject cards, table, and export buttons.
- Review Tab: Shows engine overlay (or a deterministic placeholder) with zoom.
- Dashboard Tab: Shows aggregate stats and recent evaluations (demo data).

## Backend API (FastAPI)

A lightweight REST API is available for integrating with external systems.

### Run the API server

```bash
# 1) Install dependencies
pip install -r requirements.txt

# 2) Start the server (hot reload)
uvicorn fastapi_app:app --reload
```

- Health check: `GET http://localhost:8000/health`
- Evaluate: `POST http://localhost:8000/evaluate`
  - Form fields (multipart):
    - `file`: PNG/JPG/JPEG/PDF upload
    - `key_set`: `A` or `B` (default `A`)

Example cURL:
```bash
curl -X POST \
  -F "file=@path/to/sheet.pdf" \
  -F "key_set=A" \
  http://localhost:8000/evaluate
```
The response JSON includes an `overlay_png_base64` field containing a base64-encoded annotated PNG.

## Database (SQLite)

Evaluation results are persisted to a local SQLite database at `omr.db` in the project root using SQLAlchemy.

- Model: `db.Result`
- Helper: `db.save_result(...)`
- The Streamlit app writes to DB automatically for real evaluations (not in Demo Mode).

To inspect the DB, you can use any SQLite browser or run Python:
```python
import sqlite3
con = sqlite3.connect('omr.db')
print(con.execute('SELECT COUNT(*) FROM results').fetchone())
```

## PDF Handling

PDF uploads are supported end-to-end:
- The UI (`app.py`) accepts PDFs.
- The engine (`engine.py`) auto-detects PDFs by header and renders the first page to an image using PyMuPDF before processing.
- Recommended: scan at 300 DPI+ for best results.

## License

This project is proprietary software developed by PaperLens. All rights reserved.

---

**Built with care by the PaperLens Team**

*Revolutionizing document processing through advanced AI*
