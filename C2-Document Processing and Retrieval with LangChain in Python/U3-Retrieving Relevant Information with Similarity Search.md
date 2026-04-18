## Retrieving Relevant Information with Similarity Search
Welcome back! In the previous lesson, we explored how to generate embeddings for document chunks using OpenAI and LangChain. Today, we will build on that knowledge by diving into vector databases and how they enable the efficient retrieval of relevant information through similarity search.
Vector databases are specialized storage systems designed to handle high-dimensional vector data, such as the embeddings we generated in the last lesson. They are crucial for performing similarity searches, which allow us to find document chunks that are semantically similar to a given query. In this lesson, we will focus on using FAISS, a powerful tool developed by Facebook AI, to create a local vector storage. This will enable us to efficiently store and search through our embeddings, paving the way for advanced document retrieval tasks.

## Preparing Documents and Embedding Model
Before we can perform a similarity search, we need to prepare our document and initialize our embedding model.

Here's a quick recap of how to do it:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Define the file path
file_path = "data/the_adventure_of_the_blue_carbuncle.pdf"

# Create a loader for our document
loader = PyPDFLoader(file_path)

# Load the document
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbeddings()
```
This code snippet demonstrates how to load a document, split it into chunks, and initialize our embedding model, preparing everything for further processing.

## Creating Embeddings and Vector Store
With our document chunks ready and embedding model initialized, the next step is to generate embeddings and create a vector store. As you learned in the previous lesson, embeddings are numerical representations of text that capture semantic meaning.

We'll use FAISS (Facebook AI Similarity Search) to create a vector store. Think of this as a specialized database designed specifically for storing and searching through embeddings efficiently.

```python
from langchain_community.vectorstores import FAISS

# Generate embeddings for all the document chunks and create a vector store
vectorstore = FAISS.from_documents(split_docs, embedding_model)
```

This single line of code handles all the complex work of embedding generation and storage, making it easy for us to perform similarity searches in the next step. Let's break down what's happening in this code:

1. We import the FAISS class from LangChain's vector store collection.
2. We call FAISS.from_documents() and pass two important parameters:
* split_docs: Our list of document chunks that we want to search through later.
* embedding_model: Our OpenAI embedding model that will convert each text chunk into a vector.
Behind the scenes, this method:

* Takes each document chunk from split_docs.
* Uses the embeddings model to convert each chunk's text into a numerical vector.
* Organizes all these vectors in the FAISS index for efficient searching.
* Returns a ready-to-use vector store that maintains the connection between the vectors and their original text.
* It’s worth noting that the association between the embedding vectors and the original document objects (including their metadata) is preserved within the vector store. This is important because it enables the system not just to retrieve matching text chunks, but also to surface metadata like the page number or source file—critical in multi-document applications or user-facing interfaces.

## Performing Similarity Search
Now that we have our vector store, we can perform a similarity search to retrieve relevant documents. Similarity search involves finding document chunks whose embeddings are closest to a given query's embedding. This allows us to extract information that is semantically similar to the query.

Here's how we perform a similarity search:

```python
# Define our search query
query = "What was the main clue?"

# Perform similarity search to find the top 3 most relevant document chunks
retrieved_docs = vectorstore.similarity_search(query, k=3)

# Loop through each retrieved document
for doc in retrieved_docs:
    # Print the first 300 characters of each document chunk
    print(doc.page_content[:300], "...\n")
```
When we run this code with our Sherlock Holmes story, we get the following output:
```text
The little man stood glancing from one to the
other of us with half-frightened, half-hopeful eyes,
as one who is not sure whether he is on the verge
of a windfall or of a catastrophe. Then he stepped
into the cab, and in half an hour we were back in
the sitting-room at Baker Street. Nothing had been ...

less innocent aspect. Here is the stone; the stone
came from the goose, and the goose came from Mr.
Henry Baker, the gentleman with the bad hat and
all the other characteristics with which I have bored
you. So now we must set ourselves very seriously
to ﬁnding this gentleman and ascertaining what
pa ...

she found matters as described by the last
witness. Inspector Bradstreet, B division,
gave evidence as to the arrest of Horner,
who struggled frantically, and protested his
innocence in the strongest terms. Evidence
of a previous conviction for robbery having
been given against the prisoner, the mag ...
```

As you can see, the similarity search has retrieved three document chunks that are semantically related to our query about the "main clue" in the story. Even though the exact phrase "main clue" might not appear in the text, the system has identified passages that discuss evidence, the stone (the blue carbuncle), and the investigation - all relevant to our query about clues in the mystery.

## Summary and Next Steps
In this lesson, you learned how to create a local vector storage with FAISS and perform a similarity search to retrieve relevant information from documents. We built on your knowledge of document loading, splitting, and embedding to enable efficient document retrieval.

As you move on to the practice exercises, I encourage you to experiment with different documents and queries to solidify your understanding. This hands-on practice will prepare you for the next unit, where we will continue to build on these skills. Keep up the great work, and I look forward to seeing you in the next lesson!

