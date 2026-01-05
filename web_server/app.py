import streamlit as st


#MODEL_NAME =
st.set_page_config(page_title="Veterinary Document Processing")

st.header("Veterinary Document Processing")
st.markdown("Web site for securely processing medical documents with AI.")

#   Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

# Upload file (need to check filetypes supported)
uploaded_file = st.file_uploader(
    label="Upload file",
    type=["jpeg", "pdf", "jpg", "png"], )

# Placeholder for OCR model call
def get_model_response(file_bytes, filename):
    '''

    :param file_bytes:
    :param filename:
    :return:
    '''

    #TODO: Replace with actual model call
    return f"{filename} successfully processed"

#   Process File
if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    with st.spinner("Processing document..."):
        ocr_text = get_model_response(file_bytes, uploaded_file.name)

        #Save response to history
    st.session_state.history.append({
        "filename" : uploaded_file.name,
        "output"   : ocr_text
    })

    #   Display result
    st.subheader("OCR output")
    st.write(ocr_text)

#   Display History
if st.session_state.history:
    st.subheader("Session History")
    for item in st.session_state.history:
        st.markdown(f"**{item['filename']}**")
        st.write(item["output"])
        st.markdown("---")



