import streamlit as st

def stack_page():
    st.title('Technology Stack')
    cols = st.columns(5)

    logos = {
        cols[0]: 'logo/langchain.jpg',
        cols[1]: 'logo/ollama.jpg',
        cols[2]: 'logo/flutter.png',
        cols[3]: 'logo/fastapi.png',
        cols[4]: 'logo/streamlit.png',
    }

    for col, img_path in logos.items():
        col.image(img_path, use_container_width=True)

if __name__ == "__main__":
    stack_page()
