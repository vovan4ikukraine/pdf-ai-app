from pathlib import Path
import pdfplumber
import chromadb
from openai import OpenAI
import json
import re
import os
from dotenv import load_dotenv

# ---------------------- Функции ----------------------


def extract_pdf_text(pdf_path: str) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path.resolve()}")

    pages_text = []
    with pdfplumber.open(path) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.replace("\xa0", " ").strip()
            pages_text.append(
                f"\n\n=== PAGE {idx}/{len(pdf.pages)} ===\n{text}")
    return "\n".join(pages_text).strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def save_text(text: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Сохранён txt: {out_path}")

# ---------------------- Настройка клиентов ----------------------


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("all_cvs")

# ---------------------- Индексация PDF ----------------------

PDF_FOLDER = Path("pdfs")        # папка с PDF
TXT_FOLDER = Path("pdf_txts")    # папка для txt

# Сохраняем список всех id в коллекции (для проверки)
existing_ids = set(collection.get()["ids"])

for pdf_file in PDF_FOLDER.glob("*.pdf"):
    pdf_id_prefix = pdf_file.stem

    # проверяем, есть ли уже этот PDF в коллекции
    already_indexed = any(id_.startswith(pdf_id_prefix)
                          for id_ in existing_ids)
    if already_indexed:
        print(f"⚡ PDF {pdf_file.name} уже в базе, пропускаем.")
        continue

    else:
        print(f"📄 Обрабатываем новый PDF: {pdf_file.name}...")
        text = extract_pdf_text(pdf_file)

        # сохраняем txt
        txt_file = TXT_FOLDER / f"{pdf_file.stem}.txt"
        save_text(text, txt_file)

    # создаём embeddings
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        embedding = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding
        collection.add(
            ids=[f"{pdf_file.stem}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"pdf_name": pdf_file.name, "chunk_index": i}]
        )

print("✅ Все PDF обработаны и добавлены в коллекцию.")


# ---------------------- Функция запроса ----------------------

chat_history = []  # здесь будем хранить историю (вопросы и ответы)


def rerank_and_answer(question: str,
                      candidates: int = 10,
                      top_k: int = 3,
                      reranker_model: str = "gpt-3.5-turbo",
                      answer_model: str = "gpt-4o-mini"):
    """
    1) Берём `candidates` из Chroma (по embedding вопроса).
    2) Просим LLM (reranker_model) выбрать top_k индексов (0-based) в виде JSON списка.
    3) Формируем context из выбранных кандидатов и запрашиваем окончательный ответ у answer_model.
    Возвращает (answer, debug) где debug содержит candidates, reranker_raw, selected_indices, final_context.
    """

    # 1. эмбеддинг вопроса и поиск кандидатов
    qemb = client.embeddings.create(
        model="text-embedding-3-small", input=question)
    qvec = qemb.data[0].embedding

    results = collection.query(
        query_embeddings=[qvec],
        n_results=candidates,
        include=["documents", "metadatas", "distances"]
    )

    cand_texts = results.get("documents", [[]])[0]
    cand_metas = results.get("metadatas",  [[]])[0]
    cand_dists = results.get("distances",  [[]])[0]

    # безопасность: если нет кандидатов — сразу fallback
    if not cand_texts:
        return "Ничего не найдено в базе.", {"error": "no_candidates"}

    # 2. Формируем префикс с нумерацией — короткие превью для reranker'а (чтобы не присылать огромные тексты)
    #    Показываем первые 800 символов каждого кандидата (чтобы реранкер понимал суть).
    preview_max = 800
    enumerated = []
    for i, txt in enumerate(cand_texts):
        preview = txt if len(txt) <= preview_max else txt[:preview_max].rsplit(" ", 1)[
            0] + "..."
        meta = cand_metas[i] if i < len(cand_metas) else {}
        source = meta.get("pdf_name", "unknown")
        enumerated.append(f"{i}: [{source}] {preview}")

    candidates_block = "\n".join(enumerated)

    # 3. Промпт для рерангера — просим вернуть JSON списка индексов (0-based)
    rerank_prompt = f"""
У тебя есть ВОПРОС:
{question}

Ниже список кандидатов (размечен индексами слева). Каждый элемент — текстовый фрагмент (preview) с источником.
Задача: выбери {top_k} наиболее релевантных фрагментов, которые помогут ответить на вопрос.
ОТВЕЧАЙ ТОЛЬКО JSON МАССИВОМ ИНДЕКСОВ (0-based), например: [0, 3, 7]
Ни в коем случае не добавляй пояснений — только рабочий JSON.

Кандидаты:
{candidates_block}
"""

    # 4. Запрос к reranker (маленькая модель — дешевле)
    rerank_resp = client.chat.completions.create(
        model=reranker_model,
        messages=[
            {"role": "system", "content": "Ты помогаешь выбрать наиболее релевантные фрагменты текста."},
            {"role": "user", "content": rerank_prompt}
        ],
        temperature=0.0,  # детерминированность лучше для парсинга
    )
    rerank_raw = rerank_resp.choices[0].message.content.strip()

    # 5. Пытаемся распарсить JSON из ответа
    selected_indices = None
    try:
        # Пытаемся найти JSON array в тексте
        m = re.search(r"(\[.*\])", rerank_raw, re.DOTALL)
        if m:
            candidate_json = m.group(1)
            selected_indices = json.loads(candidate_json)
            # нормализуем (убедимся, что это список int)
            selected_indices = [int(x) for x in selected_indices][:top_k]
    except Exception:
        selected_indices = None

    # 6. Фоллбек: если парсинг не удался — используем расстояния (меньше distance == ближе)
    if not selected_indices:
        # сортируем по distance и берём первые top_k индексов
        try:
            idxs_sorted = sorted(range(len(cand_dists)),
                                 key=lambda i: cand_dists[i])
            selected_indices = idxs_sorted[:top_k]
        except Exception:
            # если нет distances — просто берем первые top_k
            selected_indices = list(range(min(top_k, len(cand_texts))))

    # 7. Формируем финальный контекст из выбранных chunks (и метаданных)
    final_parts = []
    for i in selected_indices:
        meta = cand_metas[i] if i < len(cand_metas) else {}
        source = meta.get("pdf_name", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        final_parts.append(f"[{source} | chunk {chunk_idx}]\n{cand_texts[i]}")

    final_context = "\n\n---\n\n".join(final_parts)

    # 8. Финальный запрос к LLM — ответ на вопрос, используя только этот контекст
    system_msg = {
        "role": "system",
        "content": "Ты ассистент. Ответь только на основе предоставленного контекста. Если информации недостаточно — честно скажи."
    }
    user_msg = {
        "role": "user",
        "content": f"Контекст:\n{final_context}\n\nВопрос: {question}"
    }

    answer_resp = client.chat.completions.create(
        model=answer_model,
        messages=[system_msg, user_msg],
        temperature=0.0
    )
    answer = answer_resp.choices[0].message.content

    debug = {
        "candidates_count": len(cand_texts),
        "candidates_preview": enumerated,
        "reranker_raw": rerank_raw,
        "selected_indices": selected_indices,
        "distances": cand_dists
    }

    return answer, debug

# ---------------------- Интерактивный режим ----------------------


if __name__ == "__main__":
    while True:
        user_input = input("Введите вопрос ИИ (или 'выход' для завершения): ")
        if user_input.lower() == "выход":
            print("Выход из программы.")
            break
        answer, debug = rerank_and_answer(user_input, candidates=10, top_k=3)
        print("\n=== DEBUG ===")
        print("Reranker raw:\n", debug["reranker_raw"])
        print("Selected indices:", debug["selected_indices"])
        print("\n=== ANSWER ===\n", answer)
