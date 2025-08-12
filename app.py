import streamlit as st
from main_page import show_main_page
from oni import show_main_page3
from siyo import siyo

# Secretsから読み込み
USER_CREDENTIALS = st.secrets["users"]

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

def login():
    st.title("ログインページ")
    username = st.text_input("ユーザー名")
    password = st.text_input("パスワード", type="password")

    if st.button("ログイン"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("ログイン成功！")
            st.rerun()
        else:
            st.error("ユーザー名またはパスワードが違います")

# 実行
if st.session_state.logged_in:
    if st.session_state.username == "y":
        show_main_page()
    elif st.session_state.username == "o":
        show_main_page3()
    elif st.session_state.username == "s":
        siyo()
    else:
        st.error("このユーザーには対応するページがありません")
else:
    login()

