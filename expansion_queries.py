from helper_utils import project_embeddings, word_wrap
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import os
import umap
import numpy as np
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import matplotlib.pyplot as plt

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_texts = [p.extract_text().strip() for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0,
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))


token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)


embedding_function = SentenceTransformerEmbeddingFunction()
# print(embedding_function([token_split_texts[10]]))

chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "microsoft-collection", embedding_function=embedding_function
)
# extract the embeddings of the token_split_texts
ids = [str(i) for i in range(len(token_split_texts))]

chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()


query = "What was the total revenue for the year?"

def generate_multiple_queries(query, model="gpt-3.5-turbo"):
    prompt = """ You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (withouth compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering. """
    messages = [
        {
            "role": "system",
            "content": prompt,  
        },
        {
            "role": "user",
            "content": query,
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

org_query = (
    "What details can you provide about the factors that led to revenue growth?"
)
aug_queries = generate_multiple_queries(org_query)
    

# show the augmented queries
# for query in aug_queries:
#     print("\n", query)

# combine the original query with the augmented queries
joint_query = [
    org_query
] + aug_queries

results = chroma_collection.query(
    query_texts=joint_query,
    n_results=5,
    include=["documents", "embeddings"],
)
retrieved_docs = results["documents"]

unique_documents = set()
for docs in retrieved_docs:
    for document in docs:
        unique_documents.add(document)


#output the results
# for i, documents in enumerate(retrieved_docs):
#     print(f"Query: {joint_query[i]}")
#     print("")
#     print("Results:")
#     for doc in documents:
#         print(word_wrap(doc))
#         print("")
#     print("-" * 100)

embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

original_query_embedding = embedding_function([org_query])
augmented_query_embeddings = embedding_function(joint_query)
retrieved_embeddings = results["embeddings"]
result_embeddings = [item for sublist in retrieved_embeddings for item in sublist]

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embeddings = project_embeddings(
    augmented_query_embeddings, umap_transform
)
projected_result_embeddings = project_embeddings(
    result_embeddings, umap_transform
)

plt.figure()
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_augmented_query_embeddings[:, 0],
    projected_augmented_query_embeddings[:, 1],
    s=150,
    marker="X",
    color="orange",
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{org_query}")
plt.axis("off")
plt.show() 