
import os
from pathlib import Path
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.qdrant_retrieve_user_proxy_agent import QdrantRetrieveUserProxyAgent,RetrieveUserProxyAgent
from autogen.agentchat.contrib.retrieve_assistant_agent import AssistantAgent,RetrieveAssistantAgent
from loading import Loader
from langchain_experimental.text_splitter import SemanticChunker


load_dotenv(Path("../api_key.env"))

def main():
      
    
    config_list=[{
        "model": "gpt-3.5-turbo",
        "api_key": os.environ.get("OPENAI_API_KEY"),
    }]

    llm_config={
            "timeout": 600,
            "cache_seed": 42,
            "config_list": config_list,
            "cache_root_path": "./custom_cache_path"
        }
    

    EVAL_CUSTOM_PROMPT =  """Sei un valutatore di qualità della risposta di un chatbot per una compagnia di assicurazioni.
    In base alla domanda ed al contesto genera una label tra le seguenti: [CORRETTO,ERRATO].

    Rispondi nel seguente formato: " --LABEL-- --spiegazione scelta label-- --se erratta,riscrivi la risposta correttamente-- ".

    La domanda dell'utente è: ""{input_question}"".

    Il contesto è: ""{input_context}"".

    La risposta è: ""{input_answer}"".
    """


     
    files_path = "/home/utente/Desktop/Projects/CHATBOT_YOLO/ALL_FILES/COMPANY/SUB"
    

    RETRIEVER_CUSTOM_PROMPT = (
    "Sei un assistente chatbot per una compagnia di assicurazioni (Yolo-insurance). "
    "Usa i seguenti pezzi di contesto recuperato per rispondere "
    "alla domanda. Quando citi la nostra compagnia di assicurazione parla in "
    "prima personale plurale (e.g contatta la nostra assistenza clienti)."
    "Non citare mai che hai usato il contesto"
    " o documenti nella risposta "
    " e non citare mai riferimenti ad altre compagnie assicurative."
    "Se non conosci la risposta o non trovi "
    "la risposta nel contesto,scusati e rispondi semplicemnte di contattare l'assistenza"
    "\n"
    "La domanda dell'utente è :""{input_question}"" \n"
    "Contesto: ""{input_context}"" "
    )

    ragproxyagent = QdrantRetrieveUserProxyAgent(

        name="admin",
        system_message = """RAG answer generator,given a question and a context,
        answer the question relying on the context. 
        Then pass to the evaluator""",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        retrieve_config={
            "docs_path":files_path,
            "task": "default",
            "chunk_token_size": 2000,
            #"client": client
            "vector_db": "chroma",
            "model": config_list[0]["model"],#chroma,pgvector
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "customized_prompt" : RETRIEVER_CUSTOM_PROMPT,
        },
        code_execution_config=False

    )
 

    evaluator = autogen.AssistantAgent(
        name="Evaluator",
        system_message="Evaluator. Double check the answer is correct and coeherence with the context passed,provide feeback",
        llm_config=llm_config,
    )

    corrector = autogen.AssistantAgent(
        name="Corrector",
        system_message="Corrector. Correct the answer using the feedback and the context",
        llm_config=llm_config,
    )

        
    groupchat = autogen.GroupChat(
    agents=[ragproxyagent,evaluator,corrector], messages=[], max_round=50
    )

    for agent in groupchat.agents:
        agent.reset()

    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    user_message = input("Insert prompt:""(exit to esc)"" ")

    while user_message!="exit":
        ragproxyagent.initiate_chat(
            manager,
            message=user_message
        )
        user_message = input("Insert prompt:""(exit to esc)"" ")



if __name__ == "__main__":
    main()

