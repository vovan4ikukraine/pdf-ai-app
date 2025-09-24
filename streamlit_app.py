import streamlit as st
import requests

st.set_page_config(page_title="CV Chat", page_icon="📄💬")

st.title("Поиск по CV через FastAPI")

st.write("Введите ваш вопрос и API ключ, чтобы получить ответ.")

# Поля ввода
api_key = st.text_input("API Key:", type="password")
question = st.text_input("Ваш вопрос:")

if st.button("Спросить") and question:
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,  # добавляем ключ в заголовок
    }
    payload = {"question": question}

    with st.spinner("Идёт поиск и формирование ответа..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask", json=payload, headers=headers)
            if response.status_code == 200:
                answer = response.json().get("answer")
                debug = response.json().get("debug")

                st.subheader("Ответ:")
                st.write(answer)

                with st.expander("🔍 Debug информация"):
                    st.json(debug)
            else:
                st.error(
                    f"Ошибка {response.status_code}: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"Ошибка подключения к FastAPI: {e}")
