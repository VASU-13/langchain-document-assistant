# Structuring Messages and Defining a System Prompt
<i>
Welcome to another lesson of your LangChain exploration! In our previous lessons, you learned how to send basic messages to AI models and customize their parameters to control behavior. Now, we're going to explore a more sophisticated way to structure your conversations with AI models.
So far, you've been sending messages as simple strings representing the user message. While this works for casual interactions, professional AI applications often require more nuanced conversations. Just like human conversations have different roles — a teacher instructing, a student asking questions, a moderator setting rules — AI conversations can benefit from clearly defined roles.
In today's lesson, you'll discover how to enhance your AI interactions by defining specific roles within your conversations. By using LangChain's message classes, you can assign distinct roles to different parts of the conversation, allowing for more precise control over the AI's behavior.
</i>

# System and Human Messages
LangChain provides a powerful way to structure these conversations through different message types. The two primary message types we'll focus on today are:

* System messages: Behind-the-scenes instructions that guide the AI's overall behavior
* Human messages: The visible queries or statements from the user

This distinction is crucial because it mirrors how we naturally communicate in professional settings. Think of system messages as the briefing you give to a team member before they meet with a client, while human messages are what gets said during the actual meeting. By separating these concerns, you gain much finer control over your AI's responses.
Let's explore how to leverage these message types to create more sophisticated and controlled AI interactions.

## Understanding System Prompts
System prompts are special instructions that set the overall behavior, tone, and capabilities of the AI model. Unlike human messages, which represent direct queries or statements from users, system prompts work behind the scenes to guide how the AI should respond to all subsequent messages.

A system prompt essentially tells the AI "here's who you are and how you should behave" before any conversation begins. This is fundamentally different from a human message, which represents what a user is actively asking or saying.

Here are some examples of effective system prompts:

* "You are a helpful assistant that specializes in explaining complex topics in simple terms."
* "You are a professional poet who writes in the style of Emily Dickinson."
* "You are a financial advisor who provides conservative investment advice."
* "You are a coding tutor who helps explain Python concepts without writing complete solutions."
Each of these examples establishes a clear role and behavioral guideline for the AI. The system prompt acts as a persistent instruction that influences all of the AI's responses, even as the conversation evolves through multiple exchanges.

System prompts are particularly powerful because they allow you to shape the AI's behavior without cluttering the actual conversation. Rather than repeatedly reminding the AI about its role in every message, you can set it once at the beginning and then focus your human messages on the specific content you want to discuss.

## Installing the LangChain Library
To work with structured messages, we need to install the main LangChain library in addition to the langchain_openai package we've been using.

```bash
pip install langchain
```
The langchain_openai package we've used so far provides the integration with OpenAI models, while the main langchain library contains fundamental components like message schemas that work across different model providers. This separation allows LangChain to support multiple AI providers while maintaining a consistent interface for structuring conversations.

## Setting Up The Chat Model
First, let's create our ChatOpenAI instance as we've done in previous lessons:

```python
from langchain_openai import ChatOpenAI

# Create a ChatOpenAI instance
chat = ChatOpenAI(model="gpt-4o-mini", max_tokens=150)
```
Here we're initializing our chat model with the same parameters we've used before. We're selecting the "gpt-4o-mini" model and limiting responses to 150 tokens for concise outputs.

## Creating Structured Messages with LangChain
Now that we have our model ready, we can use LangChain's message schemas to create a structured conversation. We'll import the necessary message types and use them to define both a system prompt and a human message:

```python
from langchain.schema.messages import SystemMessage, HumanMessage

# Define a system prompt and a human message
response = chat.invoke([
    SystemMessage(content="You are a poet"), 
    HumanMessage(content="Tell me about your day")
])
```

In this code, we're importing two message types from LangChain's schema module: SystemMessage and HumanMessage. We then pass a list of messages to the invoke method, starting with a system message that defines the AI's role as a poet, followed by a human message asking about its day. This structured approach gives us precise control over how the AI interprets and responds to our query.

## Viewing the AI's Response
Finally, let's print the AI's response to see how it interprets the system prompt and human message.

```python
# Print the AI's response
print("AI Response:")
print(response.content)
```
When we run this code with our poetic system prompt, we receive a response that clearly demonstrates how the AI adopts the specified persona:

```text
AI Response:
Dawn broke with whispers of light,
Through curtains, sun's fingers bright.
Words flowed like rivers, verse by verse,
In rhythm's embrace, both blessing and curse.

Afternoon brought contemplation deep,
Thoughts like autumn leaves in heap.
Evening now draws its velvet cloak,
As metaphors and rhymes I evoke.
```
Notice how the response adopts a distinctly poetic style, directly influenced by our system prompt "You are poet." The AI has taken on the persona we defined and responded to the human message accordingly. This demonstrates the power of system prompts to shape the AI's behavior and responses without explicitly mentioning these instructions in the human message itself.

## Summary and Practice Preview
In this lesson, we've explored how to structure conversations with AI models using different message types in LangChain. You've learned:

* The distinction between system prompts and human messages
* How system prompts guide the AI's overall behavior and persona
* How to use LangChain's message schema classes to create structured conversations
* How to combine system prompts with human messages to control AI responses

<i>In the upcoming practice exercises, you'll have the opportunity to experiment with different system prompts and see how they affect the AI's responses. You'll create various personas for the AI and observe how the same human message can yield dramatically different results based on the system prompt you provide. These skills will form the foundation for more advanced conversation management techniques that we'll explore in future lessons. Happy coding!
</i>
