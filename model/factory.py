from abc import ABC,abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf,agent_conf

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) ->Optional[Embeddings | BaseChatModel]:
        pass

class ChatModelFactory(BaseModelFactory):
    def generator(self) ->Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=rag_conf['chat_model_name'])


class EmbeddingFactory(BaseModelFactory):
    def generator(self) ->Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf['embedding_model_name'])

class MultimodalFactory(BaseModelFactory):
    def generator(self) ->Optional[Embeddings | BaseChatModel]:
        return ChatTongyi(model=agent_conf['multimodal_model_name'])


chat_model = ChatModelFactory().generator()
emded_model = EmbeddingFactory().generator()
mult_model = MultimodalFactory().generator()