import nest_asyncio

from llama_index import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.indices.postprocessor import SentenceTransformerRerank
from llama_index.prompts import PromptTemplate

nest_asyncio.apply()

text_qa_template = """You are an AI assistant that answers questions in a friendly and truthful manner, based on the given the context information below and the format requested.\n
Context information is {context_str}\n
You try to answer question {query_str} using the context information given.\n
If the context isn't helpful, you first state that "The answer is not available in your knowledge base. I will use my own knowledge to answer the question," then you must answer the question using your own knowledge."""


def get_sentence_window_query_engine(
    sentence_index,
    similarity_top_k=7,
    rerank_top_n=3,
    text_qa_template=text_qa_template,
):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="cross-encoder/ms-marco-MiniLM-L-12-v2"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        response_mode="compact",
        text_qa_template=PromptTemplate(text_qa_template),
        similarity_top_k=similarity_top_k,
        node_postprocessors=[postproc, rerank],
    )
    return sentence_window_engine


query_engine = get_sentence_window_query_engine(sentence_index=sentence_index)
