from metaflow import FlowSpec, step, Parameter, current, card
from metaflow.cards import Table, Markdown
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch.nn.functional as F


class EvaluateTextEmbeddingModelsFlow(FlowSpec):
    txt_embed_models = Parameter(
        "txt_embed_models",
        help="List of text embedding models",
        default=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/multi-qa-mpnet-base-cos-v1",
            "intfloat/e5-large-v2",
            "clips/mfaq",
        ],
    )

    corpus_file = Parameter(
        "corpus_file",
        help="Path to the text file containing the corpus",
        default="./data/corpus.txt",
    )
    queries_file = Parameter(
        "queries_file",
        help="Path to the text file containing the queries",
        default="./data/queries.txt",
    )

    @step
    def start(self):
        # Read corpus and queries from text files
        with open(self.corpus_file, "r") as f:
            self.corpus = [line.strip() for line in f]
        with open(self.queries_file, "r") as f:
            self.queries = [line.strip() for line in f]

        self.model_paths = self.txt_embed_models
        self.next(self.encode, foreach="model_paths")

    @step
    def encode(self):
        self.model_path = self.input
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path)

        # Check if the model is 'intfloat/e5-large-v2' and apply the appropriate encoding method
        if self.model_path == "intfloat/e5-large-v2":
            # Format data as required by the model
            self.corpus, self.queries = self.format_data_for_model()

            # Encode corpus
            self.corpus_embeddings = self.encode_sentences_v2(
                self.corpus, tokenizer, model
            )

            # Encode queries
            self.query_embeddings = [
                self.encode_sentences_v2(query, tokenizer, model)
                for query in self.queries
            ]
        elif self.model_path == "clips/mfaq":
            # Format data as required by the model
            self.corpus, self.queries = self.format_data_for_model()

            # Encode corpus
            self.corpus_embeddings = self.encode_sentences(
                self.corpus, tokenizer, model
            )

            # Encode queries
            self.query_embeddings = [
                self.encode_sentences(query, tokenizer, model) for query in self.queries
            ]
        else:
            # Format data as required by the model
            self.corpus, self.queries = self.format_data_for_model()

            # Encode corpus
            self.corpus_embeddings = self.encode_sentences(
                self.corpus, tokenizer, model
            )

            # Encode queries
            self.query_embeddings = [
                self.encode_sentences(query, tokenizer, model) for query in self.queries
            ]

        self.next(self.join)

    @step
    def join(self, inputs):
        self.corpus_embeddings = [inp.corpus_embeddings for inp in inputs]
        self.query_embeddings = [inp.query_embeddings for inp in inputs]
        self.model_paths = [inp.model_path for inp in inputs]
        self.queries = inputs[0].queries  # the queries are the same for all inputs
        self.corpus = inputs[0].corpus  # the corpus is the same for all input
        self.results = []
        self.next(self.calculate_similarity)

    @step
    def calculate_similarity(self):
        print("Inside calculate_similarity")
        self.top3_hits = []
        for model_path, corpus_embeddings, query_embeddings in zip(
            self.model_paths, self.corpus_embeddings, self.query_embeddings
        ):
            for query, query_embedding in zip(self.queries, query_embeddings):
                # Calculate cosine similarities
                cos_scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
                cos_scores = cos_scores.flatten()

                # Get top 3 document ids
                top3_doc_ids = np.argpartition(-cos_scores, range(3))[:3]
                top3_scores = cos_scores[top3_doc_ids]
                self.top3_hits.append((top3_doc_ids.tolist(), top3_scores.tolist()))

                # Print top 3 hits for each model
                for hit_id, score in zip(top3_doc_ids, top3_scores):
                    hit_text = self.corpus[hit_id]
                    self.results.append((model_path, query, hit_text, score))
                    # print(
                    #     f"Model: {model_path}, Query: {query}, Hit: {hit_text}, Score: {score}"
                    # )

        self.next(self.end)

    @card(type="blank")
    @step
    def end(self):
        data = self.results
        result = {}

        # Add top level heading for the card
        current.card.append(Markdown(f"# Embedding models evaluation"))

        # Extract query, hit_text, and score for each unique model_path
        for model_path, query, hit_text, score in data:
            if model_path not in result:
                result[model_path] = []
            result[model_path].append(
                {"query": query, "hit_text": hit_text, "score": score}
            )
        # print(result)
        # Print the extracted data for each unique model_path
        for model_path, entries in result.items():
            print(f"Model Path: {model_path}")
            current.card.append(Markdown(f"## Model = {model_path}"))
            rows = []
            for entry in entries:
                query = entry["query"]
                hit_text = entry["hit_text"]
                score = "{:.4f}".format(entry["score"])
                # print(f"Query: {query}, Hit: {hit_text}, Score: {score}")entry["score"]
                rows.append(
                    [
                        Markdown(f"**{query}**"),
                        Markdown(f"*{hit_text}*"),
                        Markdown(f"**{score}**"),
                    ]
                )
            headers = ["Query", "hit_text", "cosine_similarity_score"]
            current.card.append(Table(rows, headers))

    def format_data_for_model(self):
        """
        Format the data for the card

        clips/mfaq:
        You can use MFAQ with sentence-transformers or directly with a HuggingFace model.
        In both cases, questions need to be prepended with <Q>, and answers with <A>.

        intfloat/e5-large-v2:
        Each input text should start with "query: " or "passage: ".
        For tasks other than retrieval, you can simply use the "query: " prefix.
        """
        # Check if the model is 'intfloat/e5-large-v2' and apply the appropriate encoding method
        if self.model_path == "intfloat/e5-large-v2":
            #  Each input text should start with "query: " or "passage: ".
            #  For tasks other than retrieval, you can simply use the "query: " prefix.
            corpus = []
            queries = []
            for i in range(len(self.corpus)):
                corpus.append(f"passage: {self.corpus[i]}")
            for i in range(len(self.queries)):
                queries.append(f"query: {self.queries[i]}")
        elif self.model_path == "clips/mfaq":
            # You can use MFAQ with sentence-transformers or directly with a HuggingFace model.
            # In both cases, questions need to be prepended with <Q>, and answers with <A>.
            corpus = []
            queries = []
            for i in range(len(self.corpus)):
                corpus.append(f"<A>{self.corpus[i]}")
            for i in range(len(self.queries)):
                queries.append(f"<Q>{self.queries[i]}")
        else:
            # Use the default encoding method
            corpus = self.corpus
            queries = self.queries

        # print(f"queries: {queries}")
        return corpus, queries

    def encode_sentences(self, sentences, tokenizer, model, normalize=True):
        # Tokenize sentences
        encoded_input = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling. In this case, mean pooling
        sentence_embeddings = self.mean_pooling(
            model_output, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def encode_sentences_v2(self, sentences, tokenizer, model, normalize=True):
        # Tokenize sentences
        encoded_input = tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform average pooling
        sentence_embeddings = self.average_pool(
            model_output.last_hidden_state, encoded_input["attention_mask"]
        )

        # Normalize embeddings
        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def average_pool(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


if __name__ == "__main__":
    EvaluateTextEmbeddingModelsFlow()
