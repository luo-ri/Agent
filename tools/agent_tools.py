from rag.rag_service import RagSummarizerService
from langchain_core.tools import tool
from utils.logger_handler import logger

rag_herb = RagSummarizerService('herb')
rag_prescription = RagSummarizerService('prescription')
rag_symptom = RagSummarizerService('symptom')



@tool(description='查询中药材的性味归经功效信息，入参为药材名称（如"知母""黄芪"）')
def get_herbs(query:str) -> str:
    return rag_herb.rag_summarizer(query)

@tool(description='查询中医方剂的组成功效主治，入参为方剂名称（如"六味地黄丸"）')
def get_Prescription(query:str) -> str:
    return rag_prescription.rag_summarizer(query)

@tool(description='基于患者症状进行中医辨证分析，入参为症状描述（如"口干舌燥,潮热,盗汗"）')
def get_symptoms(query:str) -> str:
    return rag_symptom.rag_summarizer(query)