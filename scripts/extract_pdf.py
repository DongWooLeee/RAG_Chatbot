import os
import pypdfium2

import pypdfium2

def extract_pdf_text(pdf_path: str) -> str:
    """pypdfium2를 사용하여 PDF에서 텍스트 추출"""
    pdf = pypdfium2.PdfDocument(pdf_path)  # PDF 로드
    text = ""
    
    for i in range(len(pdf)):  # 페이지 별로 탐색
        page = pdf[i]
        text_page = page.get_textpage()
        text += text_page.get_text_range() + "\n"  # 페이지 별로 텍스트 추출

    pdf.close()  # ✅ 명시적으로 PDF 객체 닫기 (메모리 해제)
    return text.strip()


def extract_pdf(pdf_path: str):
    """✅ 단일 PDF 파일에서 텍스트를 추출하는 함수"""
    text = extract_pdf_text(pdf_path)
    return [{"name": pdf_path, "content": text}]


def extract_pdfs_from_folder(folder_path: str):
    """ 폴더 내 모든 PDF 파일을 텍스트로 변환 """
    documents = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            text = extract_pdf_text(pdf_path)
            documents.append({"name": file, "content": text})
    return documents


