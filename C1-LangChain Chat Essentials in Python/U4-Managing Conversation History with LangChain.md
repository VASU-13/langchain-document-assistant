# Managing Conversation History with LangChain

Welcome to your next lesson on LangChain! So far, you've learned how to send basic messages to AI models, customize their parameters, and structure conversations using SystemMessage and HumanMessage. These skills have given you the foundation to create single-turn interactions with AI models. However, real-world conversations rarely consist of just one exchange.

Think about how you communicate with friends or colleagues. You ask a question, they respond, and then you might ask a follow-up question based on their answer. This natural flow of conversation relies on both participants remembering what was previously discussed. Without this shared context, conversations would feel disjointed and repetitive. The same principle applies when building AI applications. To create truly engaging and helpful AI assistants, we need to maintain conversation history across multiple exchanges. This allows the AI to understand references to previous messages and provide coherent, contextually appropriate responses.

In this lesson, we'll build on your knowledge of message types to implement multi-turn conversations. You'll learn how to create and manage a persistent conversation history, enabling the AI to maintain context across multiple exchanges. By the end of this lesson, you'll be able to create a conversational AI that can remember previous exchanges and respond appropriately to follow-up questions. Let's get started!

## Working with Message Lists in LangChain

In our previous lesson, we learned how to use SystemMessage and HumanMessage classes to structure a single exchange with an AI model. Now, we'll expand on this concept by working with lists of messages that persist throughout a conversation.

The key to managing conversation history in LangChain is to maintain a list of messages that grows as the conversation progresses. Each message in this list represents a turn in the conversation, whether it's a system instruction, a human query, or an AI response.

Let's start by creating a message list to store our conversation:

```python
from langchain.schema.messages import SystemMessage, HumanMessage

# Define initial messages for the conversation
messages = [
    SystemMessage(content="You are a math assistant"),
    HumanMessage(content="What is the square root of 9?")
]

```
In this code, we're creating a list called messages that contains our initial conversation state. We start with a system message that defines the AI's role as a math assistant, followed by a human message asking about the square root of 9. This list will serve as the foundation for our ongoing conversation.

## Getting the First Response
Now that we have our initial messages list, let's send it to the AI model to get a response:

```python
from langchain_openai import ChatOpenAI

# Create a ChatOpenAI instance
chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=50)

# Send the initial messages to the AI model
response = chat.invoke(messages)

# Display the first AI response
print(f"First Response: {response.content}")
```


When we run this code, we'll see output similar to:
```text
First Response: The square root of 9 is 3.
```

Notice that we're passing the entire messages list to the invoke method, not just the human message. This allows the AI to see both the system instruction (that it should act as a math assistant) and the human query (about the square root of 9). 
The AI then generates a response based on this complete context.

## Updating the Conversation History
To create a true multi-turn conversation, we need to add the AI's response to our message history and then ask a follow-up question. This is where the AIMessage class comes into play.

After receiving the AI's response, we can add it to our message list using the AIMessage class:

```python
from langchain.schema.messages import AIMessage

# Add the AI's response to the conversation history
messages.append(AIMessage(content=response.content))
```

This line takes the content of the AI's response and wraps it in an AIMessage object, which we then append to our messages list. Our conversation history now contains three messages: the system instruction, the human's first question, and the AI's response.

With the AI's response added to our conversation history, we can now ask a follow-up question:

```python
# Add a new human message to the conversation
messages.append(HumanMessage(content="And 16?"))
```
This line adds a new human message to our conversation history. Notice that the message is quite brief: "And 16?" In a normal conversation without history, this would be too vague for the AI to understand. However, because we're maintaining conversation history, the AI will have the context to understand that we're asking about the square root of 16.

# Getting the Next Response
Now that we've updated our conversation history with both the AI's first response and our follow-up question, let's send the updated messages list to the AI model:

```python
# Send the updated conversation to the AI model
response = chat.invoke(messages)

# Display the second AI response
print(f"Second Response: {response.content}")
```

It's important to note that we're passing the entire messages list to the invoke method again, not just the new human message. This list now contains four messages: the system instruction, the first human question, the AI's first response, and the follow-up question. By sending the complete conversation history, we ensure the AI has all the context it needs to provide a relevant response.
When we run this code, we'll see output similar to:

```python
Second Response: The square root of 16 is 4
```

The AI correctly interprets our follow-up question as asking for the square root of 16, even though we didn't explicitly mention "square root" in our second message. This demonstrates the power of maintaining conversation history: the AI can understand context and references to previous exchanges, enabling more natural and efficient communication.

# Putting It All Together
Let's put everything together to see the complete implementation of our conversational math assistant:

```python
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage, AIMessage

# Create a ChatOpenAI instance
chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=150)

# Define initial messages for the conversation
messages = [
    SystemMessage(content="You are a math assistant"),
    HumanMessage(content="What is the square root of 9?")
]

# Send the initial messages to the AI model
response = chat.invoke(messages)
print(f"First Response: {response.content}")

# Add the AI's response to the conversation history
messages.append(AIMessage(content=response.content))

# Add a new human message to the conversation
messages.append(HumanMessage(content="And 16?"))

# Send the updated conversation to the AI model
response = chat.invoke(messages)
print(f"Second Response: {response.content}")
```

This code demonstrates a complete multi-turn conversation with an AI assistant. We start with a system message and a human query, get a response from the AI, add that response to our conversation history, ask a follow-up question, and then get another response that takes into account the full conversation context.

## Summary and Practice Preview
In this lesson, you've learned how to manage conversation history in LangChain to create multi-turn interactions with AI models. Here are the key concepts we covered:

1. Using message lists to maintain conversation state across multiple exchanges
2. Incorporating the AIMessage class to capture and store AI responses
3. Adding new messages to an ongoing conversation
4. Leveraging conversation history to enable contextual understanding of follow-up questions

<i>
These techniques allow you to build more sophisticated AI applications that can engage in natural, flowing conversations with users. By maintaining conversation history, your AI assistants can understand references to previous messages, remember information shared earlier in the conversation, and provide more coherent and contextually appropriate responses.
As you work through the practice exercises, try to think about how you might apply these techniques to your own projects. Consider what types of conversations would benefit from maintained history, and how you might structure your message lists to support different conversation flows. Happy coding!
</i>

