import streamlit as st
from importlib import import_module

# Define the list of available diseases
diseases = ["diabetes", "heart", "cardiovascular", "thyroid"]

def load_disease_module(disease_name):
    try:
        # Import the module dynamically based on the disease name
        disease_module = import_module(f"diseases.{disease_name.lower()}")
        return disease_module
    except ImportError:
        return None

def main():
    st.title("Disease Portal")

    # Add a search bar
    search_query = st.text_input("Search for a disease:", "")

    # Check if a search query is entered
    if search_query:
        # Attempt to load the corresponding module based on the search query
        selected_disease_module = load_disease_module(search_query)

        # Check if the module is loaded successfully
        if selected_disease_module:
            # Display content from the selected disease module
            st.header(f"Information about {search_query}")
            selected_disease_module.display_information()
        else:
            st.warning("Disease not found. Please enter a valid disease name.")

if __name__ == "__main__":
    main()
