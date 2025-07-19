import io
import os
import pandas as pd
import pdfplumber
import camelot
import tabula
from paddlex import create_pipeline
from pdf2image import convert_from_bytes
from PyPDF2.errors import PdfReadError
from bs4 import BeautifulSoup

class UniversalTableExtractor:
    def __init__(self):
        self.paddlex_pipeline = create_pipeline(pipeline="table_recognition")

    def extract_tables(self, pdf_bytes, password=None):
        # Step 1: Try direct text extraction
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes), password=password) as pdf:
                for page in pdf.pages:
                    if page.extract_text():
                        # Try Camelot
                        try:
                            tables = camelot.read_pdf(
                                filepath_or_buffer=io.BytesIO(pdf_bytes),
                                password=password,
                                pages='all',
                                flavor='stream')
                            if tables and tables.n > 0:
                                return {'method': 'camelot', 'tables': [t.df for t in tables]}
                        except Exception:
                            pass
                        # Try Tabula-py
                        try:
                            dfs = tabula.read_pdf(
                                io.BytesIO(pdf_bytes),
                                password=password,
                                pages='all',
                                multiple_tables=True)
                            if dfs and len(dfs) > 0:
                                return {'method': 'tabula', 'tables': dfs}
                        except Exception:
                            pass
                # If no tables found, continue to OCR
        except PdfReadError as e:
            if 'password' in str(e).lower():
                return {'error': 'PDF is password protected or incorrect password.'}
            else:
                return {'error': str(e)}
        except Exception as e:
            return {'error': str(e)}

        # Step 2: OCR fallback with PaddleX
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            ocr_tables = []
            for img in images:
                temp_img_path = "temp_img.png"
                img.save(temp_img_path)
                output_gen = self.paddlex_pipeline.predict(input=temp_img_path)
                output = next(output_gen) if hasattr(output_gen, '__iter__') and not isinstance(output_gen, dict) else output_gen
                if 'table_res_list' in output:
                    for table_res in output['table_res_list']:
                        html = table_res.get('pred_html')
                        if html:
                            soup = BeautifulSoup(html, 'html.parser')
                            table_tag = soup.find('table')
                            if table_tag:
                                rows = []
                                for tr in table_tag.find_all('tr'):
                                    row = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
                                    if row:
                                        rows.append(row)
                                if rows:
                                    df = pd.DataFrame(rows[1:], columns=rows[0]) if len(rows) > 1 else pd.DataFrame(rows)
                                    ocr_tables.append(df)
            if ocr_tables:
                return {'method': 'paddlex', 'tables': ocr_tables}
            else:
                return {'error': 'No tables found in PDF (even with OCR).'}
        except Exception as e:
            return {'error': f'OCR extraction failed: {str(e)}'} 
