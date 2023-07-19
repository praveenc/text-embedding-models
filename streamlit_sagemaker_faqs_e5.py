import streamlit as st
from pathlib import Path
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import List
from tqdm import tqdm

DATAFILE = Path("./data/faqs/sagemaker_faqs.txt")

MODEL_ID = "intfloat/e5-base-v2"


class Embedder:
    def __init__(self, model_name=MODEL_ID):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def average_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def embed_passages(self, passages):
        passages = [
            f"passage: {passage}" if not passage.startswith("passage: ") else passage
            for passage in tqdm(passages, desc="Encoding passages")
        ]
        batch_dict = self.tokenizer(
            passages, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def embed_queries(self, queries):
        queries = [
            f"query: {query}" if not query.startswith("query: ") else query
            for query in tqdm(queries, desc="Encoding queries")
        ]
        batch_dict = self.tokenizer(
            queries, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def cosine_similarity(query_embeddings, passage_embeddings, topk):
    # Compute the cosine similarity between the query embeddings and passage embeddings
    similarity_scores = query_embeddings @ passage_embeddings.T

    # Get the top k hits
    topk_values, topk_indices = torch.topk(similarity_scores, topk)

    # Convert the tensor to a list
    topk_values = topk_values.tolist()
    topk_indices = topk_indices.tolist()

    # Create a list of tuples, where each tuple is (index, score)
    topk_hits = [(index, score) for index, score in zip(topk_indices, topk_values)]

    return topk_hits


# Extact questions and answers from a file of SageMaker FAQs.
# Read each line from the file, if the line starts with "Q:" then add the line to the list of questions, else, add it to answers
# question string will be prefixed with "query:" and answer string will be prefixed with "passage:" (this format is required for encoding with e5-base-v2 text embedding model)
# :param text: the contexts of the file
# :return: a tuple of list of questions and list of answers


def extract_qa(text):
    questions = []
    answers = []

    # Split text into questions/answers
    qa_pairs = re.split(r"Q:", text)

    for qa in qa_pairs[1:]:
        lines = qa.split("\n")

        # Extract question and prefix with 'query:'
        question = "query: " + lines[0].strip()
        questions.append(question)

        # Extract answer and prefix with 'passage:'
        answer = "passage: " + " ".join(lines[1:]).strip()
        answers.append(answer)

    return questions, answers

@st.cache_data
def encode_questions_answers(questions: List, answers: List, model_name: str = MODEL_ID):
    # Initialize the text embedding model
    embedder = Embedder(model_name)

    with st.spinner(f'Encoding passages with {model_name}...'):
        # corpus embeddings
        corpus_embeddings = embedder.embed_passages(answers)
        # st.write('Finished encoding passages.')

        # query embeddings
        # query_embeddings = embedder.embed_queries(questions)
        # st.write('Finished encoding.')

    return corpus_embeddings


# main function
if __name__ == "__main__":
    # read entire DATAFILE to a string
    text = DATAFILE.read_text()
    questions, answers = extract_qa(text)
    # print(f"Questions: {len(questions)}")
    # print(f"Answers: {len(answers)}")

    # Streamlit interface
    st.title("Question-Answering System with Text Embeddings")

    # Sidebar
    st.sidebar.header("Model and Parameters")
    model_name = st.sidebar.selectbox(
        "Select Model", ["intfloat/e5-base-v2", "intfloat/e5-large-v2"]
    )
    topk = st.sidebar.slider("Select Top K", 3, 5, 10)

    # Encode questions and answers with selected text embedding model
    print(f"Encoding questions and answers with {model_name} text embedding model")
    corpus_embeddings = encode_questions_answers(questions, answers, model_name)

    # Main
    query = st.text_input("Enter your question")
    if st.button("Get Top K Hits"):

        # Find the top k hits for a given query
        embedder = Embedder(model_name)
        query_embeddings = embedder.embed_queries([query])

        hits = cosine_similarity(query_embeddings, corpus_embeddings, topk)

        # Print the top k hits
        st.markdown("## Top K Hits")
        for hit in hits:
            st.markdown(f"Query: **{query}**")
            table = "| Hit Text | Score |\n| --- | --- |\n"
            for index, score in zip(hit[0], hit[1]):
                table += f"| {answers[index][9:]} | **{score:.4f}** |\n"  # remove the 'passage: ' prefix
            st.markdown(table)
