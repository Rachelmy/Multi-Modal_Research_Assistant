import streamlit as st
from unstructured.partition.pdf import partition_pdf
from PIL import Image
import os
import tempfile

def element_to_markdown(element):
    """Convert an unstructured element to Markdown."""
    category = element.category
    text = element.text.strip() if hasattr(element, "text") else ""

    if category == "Title":
        return f"# {text}\n"
    elif category == "NarrativeText":
        return f"{text}\n"
    elif category == "ListItem":
        return f"- {text}\n"
    elif category == "Table":
        return f"```\n{text}\n```\n"
    elif category == "FigureCaption":
        return f"**Figure:** {text}\n"
    elif category == "Image":
        return f"![Image]({text})\n"  # Placeholder for image path
    else:
        return f"{text}\n"

st.set_page_config(page_title="Unstructured PDF to Markdown", layout="wide")
st.title("📄 PDF Parser using `unstructured` Library")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info("Parsing PDF using `unstructured`...")

        try:
            elements = partition_pdf(
                filename=pdf_path,
                extract_images_in_pdf=True,
                extract_image_block_output_dir="images",
                infer_table_structure=True,
            )

            md_output = ""
            st.success(f"Parsed {len(elements)} elements.")

            for element in elements:
                md_piece = element_to_markdown(element)
                md_output += md_piece

            st.markdown("### 📝 Markdown Output")
            st.markdown(md_output)

            # Download button
            st.download_button("Download Markdown", md_output, file_name="output.md", mime="text/markdown")

        except Exception as e:
            st.error(f"Failed to parse PDF: {e}")
