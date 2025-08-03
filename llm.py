import os

from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()


def get_llm(
    use_tools=False,
    tools=[],
    model_name="meta-llama/Llama-3.1-8B-Instruct",
):
    """Get a Hugging Face LLM for text generation. either with or without tools."""
    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
    )

    chat_model = ChatHuggingFace(llm=llm)

    if use_tools:
        chat_model = chat_model.bind_tools(tools)

    return chat_model


if __name__ == "__main__":
    """Run the main function."""
    llm = get_llm()
    response = llm.invoke("What is the capital of France?")
    print("Response:", response.content)
