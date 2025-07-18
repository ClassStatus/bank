# To pre-download PaddleOCR models and avoid first-run download issues, run this script once or use:
#   from paddlex import create_pipeline
#   create_pipeline(pipeline="table_recognition")
# This will download models to ~/.paddlex/official_models or the default PaddleOCR cache directory.
from paddlex import create_pipeline
from pdf2image import convert_from_bytes
import numpy as np
import pandas as pd
import os

def extract_tables_from_pdf(pdf_bytes):
    """
    Extract tables from a PDF using paddlex's table_recognition pipeline.
    Args:
        pdf_bytes (bytes): PDF file content as bytes.
    Returns:
        List[pd.DataFrame]: List of tables as pandas DataFrames.
    """
    pipeline = create_pipeline(pipeline="table_recognition")
    images = convert_from_bytes(pdf_bytes, dpi=200)  # Lower DPI for faster processing
    all_tables = []
    for page_num, img_pil in enumerate(images, 1):
        print(f"Processing page {page_num} with paddlex table_recognition pipeline...")
        temp_img_path = f"temp_page_{page_num}.png"
        img_pil.save(temp_img_path)
        try:
            output = pipeline.predict(
                input=temp_img_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            # output['structure'] is a list of dicts, each representing a table
            if not output or 'structure' not in output or not output['structure']:
                print(f"No tables found on page {page_num} (paddlex predict returned empty).")
                continue
            for i, table in enumerate(output['structure']):
                # Each table['res'] is a dict with 'rec_texts' (list of row strings, tab-separated)
                if 'res' in table and 'rec_texts' in table['res']:
                    meaningful_texts = [text for text in table['res']['rec_texts'] if isinstance(text, str) and text.strip()]
                    if meaningful_texts:
                        rows = [row.split('\t') for row in meaningful_texts]
                        df = pd.DataFrame(rows)
                        all_tables.append(df)
                        print(f"Table extracted from page {page_num}, result {i+1}, rows: {len(df)}.")
                    else:
                        print(f"No meaningful text found in result {i+1} on page {page_num}.")
                else:
                    print(f"'res' or 'rec_texts' not found in result {i+1} on page {page_num}.")
        finally:
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)
    return all_tables
