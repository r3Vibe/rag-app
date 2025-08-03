import os

from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()


def get_llm(
    use_tools=False,
    tools=[],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
):
    """
    Initialize and return a Hugging Face Large Language Model for text generation.

    This function creates a chat-capable LLM using HuggingFace's inference endpoints.
    It authenticates with HuggingFace using a token from environment variables,
    configures the model parameters for optimal performance, and optionally
    binds tools for function calling capabilities.

    Args:
        use_tools (bool): Whether to enable tool/function calling capabilities.
                         Defaults to False.
        tools (list): List of tools/functions to bind to the model if use_tools=True.
                     Defaults to empty list.
        model_name (str): HuggingFace model repository ID to use.
                         Defaults to "meta-llama/Llama-3.1-8B-Instruct".

    Returns:
        ChatHuggingFace: A configured chat model ready for text generation.
                        Can be used for both streaming and non-streaming responses.

    Raises:
        Exception: If HUGGINGFACE_TOKEN is not found in environment variables
                  or if the model fails to initialize.
    """
    # Authenticate with HuggingFace using token from environment variables
    login(token=os.getenv("HUGGINGFACE_TOKEN"))

    # Create HuggingFace endpoint with specific configuration
    llm = HuggingFaceEndpoint(
        repo_id=model_name,  # Model identifier on HuggingFace Hub
        task="text-generation",  # Specify the task type
        max_new_tokens=512,  # Maximum number of tokens to generate
        do_sample=False,  # Use deterministic generation (no sampling)
        repetition_penalty=1.03,  # Slight penalty to reduce repetitive text
    )

    # Wrap the endpoint in a chat-compatible interface
    chat_model = ChatHuggingFace(llm=llm)

    # Optionally bind tools for function calling capabilities
    if use_tools:
        chat_model = chat_model.bind_tools(tools)

    return chat_model


if __name__ == "__main__":
    """
    Test the LLM functionality when script is run directly.
    
    This section demonstrates basic usage of the get_llm function by:
    1. Creating an LLM instance with default parameters
    2. Sending a simple test query
    3. Printing the response content
    
    This is useful for testing the HuggingFace connection and model functionality.
    """
    # Initialize the LLM with default settings
    llm = get_llm()

    # Test with a simple question
    response = llm.invoke("What is the capital of France?")

    # Display the model's response
    print("Response:", response.content)
