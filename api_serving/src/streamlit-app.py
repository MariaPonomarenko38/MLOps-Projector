import streamlit as st

from src.predictor import Predictor


@st.cache(hash_funcs={Predictor: lambda _: None})
def get_model() -> Predictor:
    return Predictor.default_from_model_registry()


predictor = get_model()


def single_pred():
    input_sent = st.text_input("English sentence", value="This is example input")
    if st.button("Run inference"):
        st.write("Input:", input_sent)
        intent = predictor.predict([input_sent])
        st.write("Intent:", intent)


def main():
    st.header("Intent classification")

    with st.beta_container():
        st.subheader("Single prediction")
        single_pred()

if __name__ == "__main__":
    main()