# Portfolio Rebalance Trigger

A Streamlit web application that helps portfolio managers track and manage index rebalancing events.

## Features

- Upload and parse index methodology PDFs
- Upload fund holding Excel files
- View upcoming rebalance dates
- Track fund holdings and suggested actions
- Export data for further analysis

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided URL (typically http://localhost:8501)

3. Upload your files:
   - Index Methodology PDFs (e.g., MSCI, S&P, Dow Jones)
   - Fund Holding Excel files

4. View the upcoming rebalance dates and fund holdings

## Input File Requirements

### Index Methodology PDFs
- Should contain rebalance dates in a readable format
- Supported formats: PDF

### Fund Holding Excel Files
Required columns:
- Name
- ISIN
- Weight

## Development

This application is built using:
- Streamlit for the web interface
- Pandas for data manipulation
- PyPDF2 for PDF parsing
- Openpyxl for Excel file handling

## Future Enhancements

- Email notifications for upcoming rebalances
- Calendar integration
- Portfolio simulation
- Real-time index feed integration 