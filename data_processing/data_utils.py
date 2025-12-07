import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def load_html_documents(source_dir: str) -> List[Document]:
    """
    Загружает все HTML-файлы из указанной директории.
    
    Args:
        source_dir: Путь к папке с HTML-файлами
        
    Returns:
        Список документов LangChain
    """
    source_path = Path(source_dir)
    html_files = sorted(source_path.glob("*.html"), key=lambda x: int(x.stem.split("_")[1]))
    
    all_docs = []
    for html_file in html_files:
        loader = BSHTMLLoader(
            str(html_file), 
            bs_kwargs={"features": "html.parser"}
        )
        all_docs.extend(loader.load())
    
    return all_docs


def clean_documents(docs: List[Document]) -> List[Document]:
    """
    Применяет очистку текста ко всем документам.
    
    Args:
        docs: Список документов
        
    Returns:
        Список документов с очищенным текстом
    """
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs


def clean_text(text: str) -> str:
    """
    Очищает текст от лишних символов и пробелов.
    
    Args:
        text: Исходный текст
        
    Returns:
        Очищенный текст
    """
    # Несколько переносов строк -> один
    text = re.sub(r'\n{2,}', '\n', text)
    
    # Несколько пробелов -> один
    text = re.sub(r' {2,}', ' ', text)
    
    # Убираем пробелы в начале и конце строк
    text = '\n'.join(line.strip() for line in text.split('\n'))
    
    # Убираем пустые строки в начале и конце
    text = text.strip()
    
    # Убираем неинформативные строки
    text = "\n".join(text.split("\n")[2:-3])

    return text


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Document]:
    """
    Разбивает документы на чанки для векторного поиска.
    
    Args:
        docs: Исходные документы
        chunk_size: Максимальный размер чанка в символах
        overlap: Перекрытие между соседними чанками
        
    Returns:
        Список разбитых документов
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)