from langchain_openai import ChatOpenAI, OpenAIEmbeddings

LLM_PROVIDER_IP="*.*.*.*:*"


def connect_to_llm() -> ChatOpenAI:
    return ChatOpenAI(
        base_url="http://" + LLM_PROVIDER_IP + "/v1",
        api_key="lm-studio",
        # model="google/gemma-4-12b",
        # model="tiel-coder-35b-a3b-mtp",
        model="qwen3.8-27b",
        # model="qwen3.6-27b-claude-opus-reasoning-distilled@q4_k_m",
        # model="qwen3.6-27b-claude-opus-reasoning-distilled",
        # model="qwen/qwen3-coder-30b",
        # model="nemotron-mini-4b-instruct",
        temperature=0,
        # max_tokens=4096,
        frequency_penalty = 0.3,
        profile={"max_input_tokens": 140_000},
    )


def connect_to_llm_embed() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url="http://" + LLM_PROVIDER_IP + "/v1",
        api_key="lm-studio",
        model="text-embedding-qwen3-embedding-0.6b",
        check_embedding_ctx_length=False,
    )


# docker run --name pgvector-container -e POSTGRES_USER=langchain -e POSTGRES_PASSWORD=langchain -e POSTGRES_DB=langchain -p 6024:5432 -d pgvector/pgvector:pg16