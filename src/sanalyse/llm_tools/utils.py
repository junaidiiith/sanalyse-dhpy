import tiktoken


def get_text_upto_tokens(text, limit, model="gpt-4o"):
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text into tokens
    tokens = encoding.encode(text)
    # Return the text upto the specified limit
    return encoding.decode(tokens[:limit])


def get_num_tokens(text, model="gpt-4o"):
    # Get the encoding for the specified model
    encoding = tiktoken.encoding_for_model(model)
    # Encode the text into tokens
    tokens = encoding.encode(text)
    # Return the number of tokens
    return len(tokens)
