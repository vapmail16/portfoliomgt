import streamlit as st
import pandas as pd
import PyPDF2
from datetime import datetime, timedelta
import re
from pathlib import Path
import pytz
import numpy as np
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai_api_key)

# Page setup
st.set_page_config(page_title="Portfolio Rebalance Trigger", page_icon="📊", layout="wide")

# Initialize session state
if 'index_methodologies' not in st.session_state:
    st.session_state.index_methodologies = []
if 'fund_holdings' not in st.session_state:
    st.session_state.fund_holdings = []
if 'pro_forma_data' not in st.session_state:
    st.session_state.pro_forma_data = None

# Extract rebalance rules with AI
def extract_rebalance_rules_with_ai(text, index_name):
    prompt = f"""
You are a financial data assistant. Extract the index rebalance rules from the following index methodology text. 
Return a JSON object with these fields:
- index_name: string
- frequency: string (e.g., 'quarterly', 'semi-annual', 'annual')
- rule: string (e.g., 'last business day of May and November', 'third Friday of March, June, September, December')
- months: list of months (e.g., ['May', 'November'])
- weekday_rule: string (if applicable, e.g., 'third Friday')
- notes: string (any other relevant info)

Text:
{text}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0
        )
        import json
        content = response.choices[0].message.content
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            rules = json.loads(match.group(0))
            rules['index_name'] = index_name
            return rules
        else:
            return None
    except Exception as e:
        st.error(f"OpenAI extraction error: {e}")
        return None

# Calculate next rebalance date
def calculate_next_rebalance(rule_dict):
    # Handles common patterns: 'last business day of May and November', 'third Friday of March, June, September, December'
    today = datetime.now().date()
    dates = []
    months = rule_dict.get('months', [])
    weekday_rule = rule_dict.get('weekday_rule', '').lower()
    frequency = rule_dict.get('frequency', '').lower()
    rule = rule_dict.get('rule', '').lower()
    year = today.year
    # Helper for last business day
    def last_business_day(year, month):
        from pandas.tseries.offsets import BDay
        last = pd.Timestamp(year, month+1, 1) - pd.Timedelta(days=1)
        if last.weekday() >= 5:
            last = last - BDay(last.weekday() - 4)
        return last.date()
    # Helper for nth weekday
    def nth_weekday(year, month, weekday, n):
        first = datetime(year, month, 1)
        count = 0
        for day in range(1, 32):
            try:
                d = datetime(year, month, day)
            except:
                break
            if d.weekday() == weekday:
                count += 1
                if count == n:
                    return d.date()
        return None
    # Parse rules
    if 'last business day' in rule:
        for m in months:
            month_num = datetime.strptime(m, '%B').month
            d = last_business_day(year, month_num)
            if d >= today:
                dates.append(d)
            else:
                d = last_business_day(year+1, month_num)
                dates.append(d)
    elif 'third friday' in rule:
        for m in months:
            month_num = datetime.strptime(m, '%B').month
            d = nth_weekday(year, month_num, 4, 3)  # 4=Friday
            if d >= today:
                dates.append(d)
            else:
                d = nth_weekday(year+1, month_num, 4, 3)
                dates.append(d)
    # Fallback: just return empty if not parsed
    return sorted(dates)

# Extract rebalance dates
def extract_rebalance_dates(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        pages = []
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if not isinstance(page_text, str):
                    continue
                if page_text.strip() == '':
                    continue
                if 'rebalance' in page_text.lower():
                    pages.append(page_text)
            except Exception:
                continue  # skip any page that errors
        text = "\n".join(pages)
        text = text[:10000]
        index_name = Path(pdf_file.name).stem
        rules = extract_rebalance_rules_with_ai(text, index_name)
        rebalance_dates = []
        if rules:
            rebalance_dates = calculate_next_rebalance(rules)
        return {
            'index_name': index_name,
            'dates': rebalance_dates,
            'rules': rules,
            'raw_text': text
        }
    except Exception as e:
        return f"Error reading PDF: {e}"

# Normalize columns
def normalize_columns(df):
    # Lowercase, strip whitespace, and remove special characters
    df.columns = [col.strip().lower().replace(' ', '').replace('_', '') for col in df.columns]
    return df

# Process Excel fund file with standard format (Weighting → Weight)
def process_fund_holdings(excel_file):
    try:
        df = pd.read_excel(excel_file)
        df = normalize_columns(df)
        # Accept both 'weight' and 'weighting'
        weight_col = None
        for col in df.columns:
            if col in ['weight', 'weighting']:
                weight_col = col
        required = ['name', 'isin']
        if all(col in df.columns for col in required) and weight_col:
            df = df.rename(columns={weight_col: 'weight'})
            # Robust conversion: handle % or decimal
            def parse_weight(val):
                val = str(val).strip()
                if val.endswith('%'):
                    return float(val.replace('%', '')) / 100
                try:
                    return float(val)
                except Exception:
                    return np.nan
            df['weight'] = df['weight'].apply(parse_weight)
            return df[['name', 'isin', 'weight']]
        else:
            st.warning(f"Excel file is missing required columns: 'Name', 'ISIN', 'Weight' or 'Weighting'")
            st.caption(f"Detected columns: {list(df.columns)}")
            return None
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return None

# Detect upcoming rebalance dates
def get_upcoming_rebalances(index_data, weeks=3):
    today = datetime.now(pytz.UTC)
    future = today + timedelta(weeks=weeks)
    upcoming = []
    for index in index_data:
        for date_str in index['dates']:
            try:
                date = pytz.UTC.localize(datetime.strptime(date_str, "%B %d, %Y"))
                if today <= date <= future:
                    upcoming.append({
                        'index_name': index['index_name'],
                        'date': date,
                        'days_until': (date - today).days
                    })
            except Exception:
                continue
    return sorted(upcoming, key=lambda x: x['date'])

# Map actions
def determine_action(row):
    if pd.isna(row["Action"]):
        return "HOLD"
    if row["Action"] == "REMOVE":
        return "SELL"
    if row["Action"] == "ADD" and pd.isna(row["Weight"]):
        return "BUY (New Addition)"
    if row["Action"] == "WEIGHT_CHANGE":
        try:
            if float(row["New Weight"]) > float(row["Weight"]):
                return "INCREASE"
            elif float(row["New Weight"]) < float(row["Weight"]):
                return "DECREASE"
            else:
                return "HOLD"
        except:
            return "CHECK"
    return row["Action"]

# Process pro forma file
def process_pro_forma(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        df = normalize_columns(df)
        # Accept synonyms for columns
        col_map = {}
        for col in df.columns:
            if col in ['isin']:
                col_map[col] = 'isin'
            elif col in ['action', 'indexaction']:
                col_map[col] = 'action'
            elif col in ['newweight', 'newindexweight', 'new_weight']:
                col_map[col] = 'new weight'
            elif col in ['weight', 'currentweight']:
                col_map[col] = 'weight'
        df = df.rename(columns=col_map)
        required = ['isin', 'action']
        if not all(col in df.columns for col in required):
            st.error("Pro Forma file must include at least: ISIN, Action (or IndexAction). Optional: New Weight (or NewIndexWeight)")
            st.caption(f"Detected columns: {list(df.columns)}")
            return None
        return df
    except Exception as e:
        st.error(f"Error processing Pro Forma file: {e}")
        return None

# Main app layout
def main():
    st.markdown("""
        <style>
        .centered-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .centered-table .stDataFrame, .centered-table .stTable {
            margin-left: auto !important;
            margin-right: auto !important;
        }
        .centered-header {
            text-align: center !important;
            width: 100%;
            margin: 0 auto 1rem auto;
        }
        </style>
    """, unsafe_allow_html=True)

    def center_content(content_func):
        with st.container():
            cols = st.columns([1, 2, 1])
            with cols[1]:
                content_func()

    # Centered main title
    st.markdown('<h1 class="centered-header">📊 Portfolio Rebalance Trigger</h1>', unsafe_allow_html=True)

    # --- Upload Index Methodology PDFs ---
    def pdf_section():
        st.markdown('<h2 class="centered-header">Upload Index Methodology PDFs</h2>', unsafe_allow_html=True)
        st.caption("Upload one or more index methodology PDFs. Rebalance dates will be extracted automatically.")
        if 'pdf_uploaded' not in st.session_state:
            st.session_state.pdf_uploaded = False
        if 'pdf_error' not in st.session_state:
            st.session_state.pdf_error = ''
        pdf_files = st.file_uploader("PDF files", type=['pdf'], accept_multiple_files=True, key="pdf_upload")
        pdf_error = ''
        if pdf_files:
            # Clear other section messages
            st.session_state.fund_error = ''
            st.session_state.fund_uploaded = False
            st.session_state.proforma_error = ''
            st.session_state.proforma_uploaded = False
            existing_names = {x['index_name'] for x in st.session_state.index_methodologies if isinstance(x, dict)}
            uploaded = False
            for pdf_file in pdf_files:
                pdf_name = Path(pdf_file.name).stem
                if pdf_name not in existing_names:
                    data = extract_rebalance_dates(pdf_file)
                    if isinstance(data, dict):
                        st.session_state.index_methodologies.append(data)
                        uploaded = True
                    elif isinstance(data, str) and data.startswith("Error"):
                        pdf_error = data
            if uploaded:
                st.session_state.pdf_uploaded = True
                st.session_state.pdf_error = ''
            elif pdf_error:
                st.session_state.pdf_error = pdf_error
        if st.session_state.pdf_uploaded:
            st.success("PDFs uploaded and rebalance dates extracted.")
        if st.session_state.pdf_error:
            st.error(st.session_state.pdf_error)
    center_content(pdf_section)

    st.divider()

    # --- Upload Fund Holdings ---
    def fund_section():
        st.markdown('<h2 class="centered-header">Upload Fund Holdings</h2>', unsafe_allow_html=True)
        st.caption("Upload one or more fund holding Excel files.")
        if 'fund_uploaded' not in st.session_state:
            st.session_state.fund_uploaded = False
        if 'fund_error' not in st.session_state:
            st.session_state.fund_error = ''
        excel_files = st.file_uploader("Excel files", type=['xlsx'], accept_multiple_files=True, key="excel_upload")
        fund_error = ''
        if excel_files:
            # Clear other section messages
            st.session_state.pdf_error = ''
            st.session_state.pdf_uploaded = False
            st.session_state.proforma_error = ''
            st.session_state.proforma_uploaded = False
            existing_names = {x['name'] for x in st.session_state.fund_holdings}
            uploaded = False
            for excel_file in excel_files:
                if excel_file.name not in existing_names:
                    try:
                        fund = process_fund_holdings(excel_file)
                        if fund is not None:
                            st.session_state.fund_holdings.append({
                                'name': excel_file.name,
                                'data': fund
                            })
                            uploaded = True
                    except Exception as e:
                        fund_error = f"Error reading fund file: {e}"
            if uploaded:
                st.session_state.fund_uploaded = True
                st.session_state.fund_error = ''
            elif fund_error:
                st.session_state.fund_error = fund_error
        if st.session_state.fund_uploaded:
            st.success("Fund holdings uploaded.")
        if st.session_state.fund_error:
            st.error(st.session_state.fund_error)
    center_content(fund_section)

    st.divider()

    # --- Pro Forma Upload ---
    def proforma_section():
        st.markdown('<h2 class="centered-header">Upload Pro Forma Index File</h2>', unsafe_allow_html=True)
        st.caption("Upload a pro forma index file (CSV or Excel) with ISIN and Action columns.")
        if 'proforma_uploaded' not in st.session_state:
            st.session_state.proforma_uploaded = False
        if 'proforma_error' not in st.session_state:
            st.session_state.proforma_error = ''
        uploaded_proforma = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"], key="proforma_upload")
        proforma_error = ''
        if uploaded_proforma:
            # Clear other section messages
            st.session_state.pdf_error = ''
            st.session_state.pdf_uploaded = False
            st.session_state.fund_error = ''
            st.session_state.fund_uploaded = False
            try:
                df_proforma = process_pro_forma(uploaded_proforma)
                if df_proforma is not None:
                    st.session_state.pro_forma_data = df_proforma
                    st.session_state.proforma_uploaded = True
                    st.session_state.proforma_error = ''
                    st.success("✅ Pro Forma loaded successfully.")
            except Exception as e:
                proforma_error = f"Error reading pro forma file: {e}"
                st.session_state.proforma_error = proforma_error
        elif st.session_state.proforma_uploaded:
            st.success("✅ Pro Forma loaded successfully.")
        if st.session_state.proforma_error:
            st.error(st.session_state.proforma_error)
    center_content(proforma_section)

    st.divider()

    # --- Upcoming Rebalances ---
    def rebalances_section():
        st.markdown('<h2 class="centered-header">📅 Upcoming Rebalances (AI Extracted)</h2>', unsafe_allow_html=True)
        if st.session_state.index_methodologies:
            for idx in st.session_state.index_methodologies:
                st.markdown(f"**{idx['index_name']}**", unsafe_allow_html=True)
                if idx.get('rules'):
                    with st.expander("Show extracted rules", expanded=False):
                        st.json(idx['rules'])
                if idx['dates']:
                    for d in idx['dates']:
                        days_until = (pd.to_datetime(d) - pd.to_datetime(datetime.now().date())).days
                        if 0 <= days_until <= 31:
                            st.success(f"Rebalance on {d} (in {days_until} days)")
                else:
                    st.warning("No rebalance dates found.")
        else:
            st.info("Upload index methodology PDFs to see upcoming rebalances.")
    center_content(rebalances_section)

    st.divider()

    # --- Fund Holdings Table ---
    def fund_table_section():
        st.markdown('<h2 class="centered-header">📄 Fund Holdings</h2>', unsafe_allow_html=True)
        if st.session_state.fund_holdings:
            for fund in st.session_state.fund_holdings:
                st.subheader(fund['name'], anchor=False)
                st.markdown('<div class="centered-table">', unsafe_allow_html=True)
                st.dataframe(fund['data'])
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload fund holdings Excel files to see holdings.")
    center_content(fund_table_section)

    st.divider()

    # --- Suggested Actions ---
    def actions_section():
        st.markdown('<h2 class="centered-header">🧠 Suggested Rebalance Actions</h2>', unsafe_allow_html=True)
        if st.session_state.pro_forma_data is not None and st.session_state.fund_holdings:
            for fund in st.session_state.fund_holdings:
                fund_df = fund["data"]
                merged = pd.merge(fund_df, st.session_state.pro_forma_data, on="isin", how="left")
                def determine_action(row):
                    action = row.get("action", None)
                    if pd.isna(action):
                        return "HOLD"
                    if str(action).upper() == "REMOVE":
                        return "SELL"
                    if str(action).upper() == "ADD" and pd.isna(row.get("weight", None)):
                        return "BUY (New Addition)"
                    if str(action).upper() == "WEIGHT_CHANGE":
                        try:
                            if float(row.get("new weight", np.nan)) > float(row.get("weight", np.nan)):
                                return "INCREASE"
                            elif float(row.get("new weight", np.nan)) < float(row.get("weight", np.nan)):
                                return "DECREASE"
                            else:
                                return "HOLD"
                        except:
                            return "CHECK"
                    return str(action).upper()
                merged["Suggested Action"] = merged.apply(determine_action, axis=1)
                output_cols = [col for col in ["name", "isin", "weight", "action", "new weight", "Suggested Action"] if col in merged.columns]
                output = merged[output_cols]
                actionable = output[output["Suggested Action"] != "HOLD"]
                st.subheader(f"📊 {fund['name']}", anchor=False)
                st.markdown('<div class="centered-table">', unsafe_allow_html=True)
                if not actionable.empty:
                    st.dataframe(actionable)
                    csv = actionable.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📤 Download CSV",
                        data=csv,
                        file_name=f"rebalance_actions_{fund['name'].replace('.xlsx','')}.csv",
                        mime="text/csv",
                        key=f"download_{fund['name']}"
                    )
                    st.info("All other holdings: HOLD.")
                else:
                    st.info("No actionable changes. All holdings: HOLD.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Upload both fund holdings and pro forma files to see suggested actions.")
    center_content(actions_section)

if __name__ == "__main__":
    main()
