import streamlit as st
from main_page import show_main_page

# Secretsから読み込み
USER_CREDENTIALS = st.secrets["users"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("ログインページ")
    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type="password")

    if st.button("ログイン"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.success("ログイン成功！")
            st.rerun()
        else:
            st.error("ユーザー名またはパスワードが違います")

# 実行
if st.session_state.logged_in:
    show_main_page()
else:
    login()
