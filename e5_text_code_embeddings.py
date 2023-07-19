import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def calculate_embeddings(texts, model, tokenizer, normalize=True):
    # Tokenize the texts
    text_dict = tokenizer(
        texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    # Get the embeddings for the texts
    text_outputs = model(**text_dict)
    text_embeddings = average_pool(
        text_outputs.last_hidden_state, text_dict["attention_mask"]
    )

    # Normalize embeddings if required
    if normalize:
        text_embeddings = F.normalize(text_embeddings, p=2, dim=1)

    return text_embeddings


def calculate_scores(query_embeddings, chunk_embeddings, query, chunks):
    # Calculate the similarity scores
    scores = torch.mm(query_embeddings, torch.stack(chunk_embeddings).squeeze().T) * 100

    # Get the indices of the top 3 scores
    _, indices = torch.topk(scores, 3)

    # Return the top 3 chunks and their scores
    return [(chunks[idx.item()], scores[0][idx].item()) for idx in indices[0]]


# Text to be encoded
text = """
To create a model (using Model Registry)
Model Registry is a feature of SageMaker that helps you catalog and manage versions of your model for use in ML pipelines. To use Model Registry with Serverless Inference, you must first register a model version in a Model Registry model group. To learn how to register a model in Model Registry, follow the procedures in Create a Model Group and Register a Model Version.

The following example requires you to have the ARN of a registered model version and uses the AWS SDK for Python (Boto3) to call the CreateModel API. For Serverless Inference, Model Registry is currently only supported by the AWS SDK for Python (Boto3). For the example, specify the following values:

For model_name, enter a name for the model.

For sagemaker_role, you can use the default SageMaker-created role or a customized SageMaker IAM role from Step 4 of the Prerequisites section.

For ModelPackageName, specify the ARN for your model version, which must be registered to a model group in Model Registry.


#Setup
import boto3
import sagemaker
region = boto3.Session().region_name
client = boto3.client("sagemaker", region_name=region)

#Role to give SageMaker permission to access AWS services.
sagemaker_role = sagemaker.get_execution_role()

#Specify a name for the model
model_name = "<name-for-model>"

#Specify a Model Registry model version
container_list = [
    {
        "ModelPackageName": <model-version-arn>
     }
 ]

#Create the model
response = client.create_model(
    ModelName = model_name,
    ExecutionRoleArn = sagemaker_role,
    container_list
)
"""

tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base-v2")
model = AutoModel.from_pretrained("intfloat/e5-base-v2")

# Split the text into chunks of 512 tokens
chunks = [text[i : i + 512] for i in range(0, len(text), 512)]

# Calculate embeddings for each chunk
chunk_embeddings = [calculate_embeddings(chunk, model, tokenizer) for chunk in chunks]


# Query sentence
query = "sample code to create a model?"

# Calculate embeddings for the query
query_embeddings = calculate_embeddings([query], model, tokenizer)

# Calculate and print the score
# Calculate and print the scores
top_hits = calculate_scores(query_embeddings, chunk_embeddings, query, chunks)
for i, (chunk, score) in enumerate(top_hits):
    print(
        f"Top {i+1} hit for the query: '{query}' is: '{chunk}', with a score of: {score}"
    )
