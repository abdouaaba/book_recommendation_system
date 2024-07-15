from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from app.core.config import settings

class ProcessingChain:
    def __init__(self):
        self.llm = OpenAI(openai_api_key=settings.OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY, model="text-embedding-3-large")

    def create_embeddings(self, texts):
        embeddings = self.embeddings.embed_documents(texts)

        return embeddings

    def create_faiss_index(self, df):
        if 'embeddings' not in df.columns:
            faiss_index = FAISS.from_texts(df['processed_description'].tolist(), self.embeddings, metadatas=df.to_dict('records'))
        else:
            faiss_index = FAISS.from_embeddings(zip(df['processed_description'], df['embeddings']), self.embeddings, metadatas=df.drop(columns=['embeddings'], axis=1).to_dict('records'))
        
        return faiss_index

    # if we want to rephrase user query / description
    def process_query(self, query):
        prompt_template = """
        Given the following user query,
        rephrase it to capture the essence of the book the user is looking for:

        User query: {user_query}

        Rephrased query:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["user_query"])
        
        query_chain = LLMChain(llm=self.llm, prompt=prompt)
        processed_query = query_chain.invoke(input={"user_query": query})
        return processed_query['text'].strip()
