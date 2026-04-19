# Asking Questions with Retrieved Context and Templates
Welcome to the final lesson of this course! In this lesson, we will integrate context retrieval with a chat model using LangChain. This builds on the skills you've developed in previous lessons, where you learned about document embeddings and similarity search. Today, we'll focus on using templates to format messages with extra context, enabling you to ask questions and receive answers based on the retrieved document content. This lesson will bring together all the skills you've learned so far, culminating in a comprehensive understanding of document processing and retrieval with LangChain in Python.

## Quick Reminder: Preparing Documents and Creating a Vector Store
Let's quickly recap what we've learned in previous lessons about preparing documents and creating a vector store. We'll load and prepare our document, "The Adventure of the Blue Carbuncle" and generate embeddings to create a vector store. This process is essential for effective context retrieval.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Define the file path
file_path = "data/the_adventure_of_the_blue_carbuncle.pdf"

# Create a loader for our document
loader = PyPDFLoader(file_path)

# Load the document
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# Create a vector store for all the document chunks
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embedding_model)
```
In this code, we load a document, split it into chunks, generate embeddings, and create a vector store, setting the stage for efficient context retrieval in our question-answering tasks.

## Combining Retrieved Context
Now that we have our vector store, we can integrate context retrieval with a chat model. First, we'll define a query and perform a similarity search to retrieve relevant documents based on the query. This will allow us to combine the retrieved document content to form a context for our question.
```python
# Define a query
query = "From whom was the stone stolen?"

# Retrieve relevant documents
retrieved_docs = vectorstore.similarity_search(query, k=3)

# Combine the content of retrieved documents
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
```
In this example, we define a query and retrieve the top three most relevant document chunks. The content of these chunks is combined to form a context that will be used in the next step.

## Formatting Messages with Templates
To effectively communicate with the chat model, we need to format our messages using templates. Think of a template as a fill-in-the-blank form where you can insert specific pieces of information. In our case, we want to insert the context we retrieved from the similarity search and the question we want to ask.

LangChain provides a helpful tool called ChatPromptTemplate to create these templates. It allows us to define a structure for our message, ensuring that the chat model receives all the necessary information to provide an accurate response.
```python
from langchain.prompts import ChatPromptTemplate

# Create a prompt template for RAG
prompt_template = ChatPromptTemplate.from_template(
    "Answer the following question based on the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)
```
In this code, we create a prompt template that includes placeholders for the context and the question. The ChatPromptTemplate helps us define this structure.

Notice how we're using Python's string formatting capabilities:

* We break the long string into multiple lines for better readability
* Each line is enclosed in quotes and separated by a space
* The \n characters represent line breaks in the text
* The {context} and {question} are placeholders that will be replaced with actual values
* This multi-line string approach makes our template easier to read and maintain
Next, we fill in the blanks with our retrieved context and query:
```python
# Format the prompt with our context and query
prompt = prompt_template.format(context=context, question=query)
```
Here, the context variable contains the combined content of the document chunks we retrieved using the similarity search. By inserting this context into the template, we ensure that the chat model has all the relevant information from the documents to generate a meaningful answer to our question. This way, the message we send to the chat model is complete and informative, allowing it to provide a more accurate response.
## Exploring the Formatted Prompt
Now, let's print the formatted template to see how the context and question are structured in the message.
```python
# Print the formatted prompt
print(prompt)
```
This will output something like:

```text
Human: Answer the following question based on the provided context.

Context:
the back yard and smoked a pipe and wondered
what it would be best to do.
“I had a friend once called Maudsley, who went
to the bad, and has just been serving his time in
Pentonville. One day he had met me...

Question: From whom was the stone stolen?
```
By printing the formatted template, we can verify that the context and question are correctly inserted into the template. This ensures that the chat model receives a well-structured message, allowing it to generate a relevant and accurate response.
## Asking a Question with Retrieved Context to a Chat Model
With our prompt ready, we can now move on to interacting with the chat model. We'll instantiate the chat model and use our formatted template to get a response.
```python
from langchain_openai import ChatOpenAI

# Initialize the chat model
chat = ChatOpenAI()

# Get the response from the model
response = chat.invoke(prompt)

# Print the question and the AI's answer
print(f"Question: {query}")
print(f"Answer: {response.content}")
```
In this section, we initialize the ChatOpenAI model and invoke it with our formatted prompt. The model processes the input and generates an answer based on the context and question provided. By printing both the question and the AI's answer, we can see how effectively the integration of context retrieval and templates enhances the interaction.

The output will look something like this:
```text
Question: From whom was the stone stolen?
Answer: The stone was stolen from the Countess of Morcar.
```

This demonstrates how the chat model uses the context to provide a relevant and accurate response to the question.

# Summary and Next Steps
You've successfully completed this lesson, where you learned how to integrate context retrieval with a chat model using LangChain. We explored the use of templates to format messages with extra context, allowing you to ask questions and receive answers based on the retrieved document content. This lesson consolidated all the skills you've acquired so far, equipping you with a solid understanding of document processing and retrieval with LangChain in Python. As you continue your learning journey, consider experimenting with different queries and document types to deepen your understanding. Stay tuned for the next course, where we'll build on these concepts and explore even more advanced techniques.