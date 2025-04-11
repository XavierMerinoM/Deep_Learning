import streamlit as st
from PIL import Image
from model.Model import Model
from tools.openai_int import ask_chatgpt
from tools.image_prs import attention_regions
import subprocess

# Function to control the subprocess of getting the image
def select_file():
    try:
        result = subprocess.run(["python", "./view/file_dialog.py"], capture_output=True, text=True)
        file_path = result.stdout.strip()

        return file_path
    except Exception as e:
        st.error(f"Error running subprocess: {e}")
        return None

def init():
    try:
        # Title of the app
        st.title("Ecuadorian Banana Ripeness Classifier")
        # Load the model
        model = Model()
        model_obj = model.model

        # Streamlit reruns the whole script when a button is pressed
        # Session variables are used to reload the screen to its previous state
        # Initialize session state variables
        if "file_path" not in st.session_state:
            st.session_state.file_path = ""
        if "prediction" not in st.session_state:
            st.session_state.prediction = ""
        if "latest_question" not in st.session_state:
            st.session_state.latest_question = ""
        if "latest_answer" not in st.session_state:
            st.session_state.latest_answer = ""

        # Button to ask the user to choose the image
        if st.button("Select Image"):
            # Save the image path state
            st.session_state.file_path = select_file()
            st.session_state.prediction = ""

        # Condition to control if an image was selected
        if st.session_state.file_path != "":
            st.subheader("Image perspective:")

            # Load images in screen
            image = Image.open(st.session_state.file_path)
            image_att = attention_regions(st.session_state.file_path, model_obj)
            #st.image([image, image_att], width=100, caption=['For people', 'For computers'])

            # Create two columns
            col1, col2 = st.columns(2)

            # Display images in columns
            with col1:
                st.image(image, width=300, caption='For people')
            with col2:
                st.image(image_att, width=300, caption='For computers')

            # Perform prediction if not already stored
            if not st.session_state.prediction:
                st.session_state.prediction = model.predict_single_image(st.session_state.file_path)

            st.write(f"This is a {st.session_state.prediction} banana!!!")

            st.subheader("Time for fun:")
            st.write("What would you like to know about this Banana Class?")

            question = st.text_input("Question: ", value="")

            # Ask Button (Updates Answer Without Refreshing Page)
            if st.button("Ask"):
                # Executes only if a question is sent
                if question:
                    st.session_state.latest_question = question
                    st.session_state.latest_answer = ask_chatgpt(question)

            # Display only the latest answer (Persistent)
            if st.session_state.latest_question:
                st.markdown(st.session_state.latest_answer)
        else:
            st.write("No image selected")

    except Exception as e:
        # Friendly error for the user
        #st.write("view.py - Error to load the image")
        # Error tracking for developers
        st.error(e)