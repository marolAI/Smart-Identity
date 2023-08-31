import streamlit as st
from backend.processor import process

def main():
    st.set_page_config(
        page_title="Smart Identity App",
        page_icon="🤖",
        # layout="wide"
    )
    
    st.title('Smart Identity')
    process()
    
if __name__ == '__main__':
    main()