# Generating Document Embeddings with OpenAI
Welcome back! In the previous lesson, you learned how to load and split documents using LangChain, setting the foundation for more advanced document processing tasks. Today, we will take the next step in our journey by exploring embeddings, a crucial concept in document processing.

Embeddings are numerical representations of text data that capture the semantic meaning of words, phrases, or entire documents. They are essential for working with Large Language Models (LLMs) because they allow these models to understand and process text in a meaningful way. By converting text into embeddings, we can perform various tasks such as similarity search, clustering, and classification.

In this lesson, we will focus on generating embeddings for document chunks using OpenAI and LangChain. This will enable us to enhance our document processing capabilities and prepare for context retrieval tasks in future lessons.

## Embeddings and Language Models
Embeddings play a vital role in context retrieval systems. Think of embeddings as a way to translate human language into a format that computers can understand and compare - like giving computers their own secret language decoder ring!

Imagine you have three sentences:

* "The Avengers assembled to fight Thanos"
* "Earth's mightiest heroes united against the Mad Titan"
* "My soufflé collapsed in the oven again"

Even though the first two sentences use completely different words, they're talking about the same superhero showdown. The third sentence? That's just my sad baking disaster. When we convert these sentences into embeddings (vectors of numbers), the vectors for the superhero sentences would be mathematically closer to each other than to my kitchen catastrophe.

## Context Retrieval Systems
Here's how embeddings work in a practical context retrieval system:

* <b>Document Processing</b>: First, we break down our documents into smaller chunks (like cutting a pizza into slices).
* <b>Embedding Generation</b>: We convert each chunk into an embedding vector (giving each slice its own unique flavor profile).
* <b>Storage</b>: These vectors are stored in a database or vector store (our digital pizza fridge).
* <b>Query Processing</b>: When a user asks a question, we convert that question into an embedding too.
* <b>Similarity Search</b>: We find the document chunks whose embeddings are most similar to our question's embedding (matching flavors).
* <b>Response Generation</b>: We use these relevant chunks as context for an LLM to generate an accurate answer.

For example, if you have a massive collection of movie scripts and someone asks "Who said 'I'll be back'?", the system would find and retrieve chunks with embeddings similar to the question - likely passages from Terminator scripts, even if they contain phrases like "Arnold's famous catchphrase" or "Schwarzenegger's iconic line" instead of the exact words in the query.

This powerful technique forms the foundation of modern search engines, chatbots, and question-answering systems, allowing them to understand the meaning behind words rather than just matching keywords - kind of like how your friend knows you're talking about that "one movie with the guy who does the thing" even when you're being incredibly vague!

## Document Loading and Splitting
Before we dive into generating embeddings, let's revisit the code from the previous lesson to load and split a document, and then we'll generate embeddings for the resulting chunks.
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Define the file path
file_path = "data/the_adventure_of_the_blue_carbuncle.pdf"

# Create a loader for our document
loader = PyPDFLoader(file_path)

# Load the document
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)
```

In this code, we first define the file path and choose the appropriate document loader based on the file type. We then load the document and split it into chunks using the RecursiveCharacterTextSplitter, as you learned in the previous lesson.

# OpenAI Embeddings and LangChain
LangChain makes it easy to work with various embedding models through a consistent interface. An embedding model is a specialized AI model that converts text into numerical vectors, capturing semantic meaning. When using OpenAI's embedding models, you'll access them through the same OpenAI API key you use for chat completions or other OpenAI services - no separate setup required. This integration makes it convenient to build complete AI systems using a single provider.

Let's see how to set up OpenAI embeddings with LangChain:

```python
from langchain_openai import OpenAIEmbeddings

# Initialize the OpenAI embedding model
embedding_model = OpenAIEmbeddings()
```

In this code, we import and initialize the OpenAIEmbeddings class, which connects to OpenAI's API using your API key (stored as an environment variable).

# Configuring Embedding Model Parameters
You can easily customize your OpenAI embeddings by adjusting a few simple settings. Here's how to set up your embedding model with different options:
```python
# Initialize OpenAIEmbeddings with custom parameters
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",  # Choose which embedding model to use
    dimensions=1536,                 # How detailed you want your vectors to be
    chunk_size=1000                  # How many pieces of text to process at once
)
```

Let's break down these settings:

* model: This is like choosing which brain to use for creating embeddings:

* text-embedding-3-small: A faster, lighter option that works great for most projects
* text-embedding-3-large: A more powerful option when you need extra accuracy
* text-embedding-ada-002: An older model that's still commonly used
* dimensions: Think of this as the level of detail in your embeddings. Higher numbers mean more detail but take up more storage space.

* chunk_size: When you're processing lots of text at once, this controls how many chunks to handle in each batch.

Don't worry too much about these settings when you're just starting out - the default values work perfectly fine for most projects! As you get more comfortable, you can experiment with these options to find what works best for your specific needs.

## Generating Embedding with OpenAI
Now that we have our embeddings model set up, let's generate an embedding for a document chunk:

```python
# Extract the text content from our first document chunk
document_text = split_docs[0].page_content

# Generate the embedding vector for this text
embedding_vector = embedding_model.embed_query(document_text)
```

The embed_query() method takes a text string and returns an embedding vector - a list of floating-point numbers that represents your text in a high-dimensional space. This vector captures the semantic meaning of your text, which will be essential for similarity search and other operations we'll explore in future lessons.

## Inspecting Embedding Vectors
Let's take a closer look at the embedding vector we generated. These vectors are the mathematical representation of our text in a high-dimensional space.

```python

# Print the first few elements of the embedding vector
print(f"First 5 values: {embedding_vector[:5]}")
```

When you run this code, you might see output like:

```text
First 5 values: [0.010573687963187695, -0.00014260565512813628, 0.005234466399997473, -0.02460428513586521, -0.012668783776462078]
```
Each number in the vector represents a dimension in the embedding space. These seemingly random numbers actually contain rich semantic information. The pattern of values across all dimensions captures the meaning of our text in a way that allows for mathematical comparison.

Two texts with similar meanings will have embedding vectors that are close to each other in this high-dimensional space, even if they use different words to express the same idea. For example, the embeddings for "I love pizza" and "Pizza is my favorite food" would be much closer to each other than either would be to "I need to fix my car." This mathematical representation of meaning is what makes embeddings so powerful for search, recommendation systems, and other NLP applications.

## Vector Databases for Embedding Storage
While we've generated an embedding for a single document chunk, a complete retrieval system needs to efficiently store and search through thousands or millions of these embedding vectors. This is where vector databases come into play.

Vector databases are specialized storage systems optimized for high-dimensional vector data. Unlike traditional databases that excel at exact matching, vector databases are designed for similarity search - finding vectors that are "close" to each other in mathematical space.

Popular vector database options include:

* <b>Chroma</b>: An open-source embedding database that's lightweight and easy to get started with
* <b>FAISS</b>: Facebook AI's similarity search library, known for its performance with large datasets
* <b>Pinecone</b>: A fully-managed vector database service built specifically for machine learning applications
* <b>Weaviate</b>: An open-source vector search engine with classification capabilities
These databases use sophisticated indexing techniques like Approximate Nearest Neighbors (ANN) to make similarity searches lightning-fast, even with millions of vectors. Without these specialized techniques, finding the closest vectors would require comparing a query against every single vector in your collection - computationally impossible for large-scale applications.

In upcoming lessons, we'll explore how to use LangChain's integrations with these vector databases to build efficient retrieval systems that can quickly find the most relevant document chunks for any query.


## Summary and Next Steps
In this lesson, you learned how to generate embeddings for document chunks using OpenAI and LangChain. We discussed the importance of embeddings in NLP and their role in context retrieval systems. You also saw a practical example of generating and inspecting embedding vectors.

As you move on to the practice exercises, you'll have the opportunity to apply these concepts by generating different embeddings and exploring their properties. This will reinforce your understanding and prepare you for the next unit, where we'll focus on retrieving relevant information using similarity search. Keep up the great work!
