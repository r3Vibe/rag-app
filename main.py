from graph import create_rag_agent


def main():
    """ " main function to run the application."""

    rag_agent = create_rag_agent()

    while True:
        user_query = input("Enter your query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        # response = rag_agent.stream(input={"query": user_query})

        # print("Response:", response["answer"])

        for token, metadata in rag_agent.stream(
            input={"query": user_query}, stream_mode="messages"
        ):
            print(token.content, end="", flush=True)


if __name__ == "__main__":
    """Run the main function."""
    main()
