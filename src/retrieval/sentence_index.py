import openai
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceWindowNodeParser

from src.utils.get_keys import get_openai_api_key

openai.api_key = get_openai_api_key()

INDEX_DIR = Path("./data/final/sentence_index")

llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, max_tokens=500)

embedding = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L12-v2")


def build_sentence_window_index(
    documents: List[Document],
    llm,
    embed_model,
    index_dir: Path | str,
    window_size: int = 3,
) -> VectorStoreIndex:
    """
    Get the sentences and context around them from the list of documents and build a sentence index.

    Args:
        documents (List[Document]):
            list of documents to extract sentences from
        window_size (int, default = 3):
            number of sentences to include around the target sentence,
            defaults to 3, meaning 1 sentence before and 1 sentence after.
        llm:
            language model to use for sentence generation
        embed_model (HuggingFaceEmbedding, default = sentence-transformers/all-MiniLM-L12-v2):
            embedding model to use for sentence similarity
        index_dir (Path | str):
            directory to save the index to
    """

    # create the sentence window node parser w/ default settings
    sent_context_parser = SentenceWindowNodeParser.from_defaults(
        window_size=window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )

    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=sent_context_parser,
    )

    # if the index doesn't exist, build it
    if not index_dir.exists():
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )

        sentence_index.storage_context.persist(persist_dir=index_dir)
    # otherwise, load it from storage
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_dir),
            service_context=sentence_context,
        )
    return sentence_index


sentence_index = build_sentence_window_index(
    md_docs, window_size=3, index_dir=INDEX_DIR
)
