import os
import tempfile
from importlib import import_module
import re
import openai
import uvicorn
import tempfile
import urllib.parse
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import pdfplumber
import boto3
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from azure.storage.blob import BlobServiceClient
from langchain.vectorstores.pgvector import PGVector
from langchain.vectorstores.pgvector import DistanceStrategy
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from fastapi.staticfiles import StaticFiles
import asyncio
import uvicorn
import boto3, json, os
from dotenv import load_dotenv


def run_main_script():
    app = FastAPI()
    security = HTTPBasic()
    container_name = "sample"

    load_dotenv()

    DELIMITER = "\n* "
    MAX_SECTION_LENGTH = 1500
    TEMPERATURE = 0.3
    MAX_TOKENS = 100
    TOP_P = 0.7
    FREQUENCY_PENALTY = 0
    PRESENCE_PENALTY = 0
    BEST_OF = 1
    STOP = None
    USER_CREDENTIALS = {"username": "admin", "password": "admin"}
    openai.api_type = "azure"
    # openai.api_version = "2022-12-01"
    openai.api_version = "2023-03-15-preview"
    openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
    openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host="vector-1.c6snayjzs6va.us-east-1.rds.amazonaws.com",
        port="5432",
        database="postgres",
        user="postgres",
        password="oHBDfZ8lLmg8atgFQPih"
    )

    def create_bedrock_session():

        access_key = os.getenv("ACCESS_KEY")
        secret_key = os.getenv("SECRET_KEY")

        bedrock = boto3.client(
            service_name='bedrock-runtime',  # change service name to 'bedrock' if you want to list foundation models
            region_name='us-east-1',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        return bedrock

    active_session = create_bedrock_session()

    chat_history = []

    prompt_template = """You are an assistant for an intelligent chatbot designed to help users answer medical questions.
    Instructions:/n
    - Only answer questions related to medical literature.
    - Answer truthfully and as human like as possible.
    - Provide your answer in 2 to 3 sentences.
    - If the answer is not contained within the text below or you are unsure of an answer, you can say "Could not find the relevant information. Please try again by rephrasing your question."./n
    
    Context: {context}
    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}
    VECTOR_STORE_PATH = "./data/roche/storage/"

    #"anthropic.claude-v2"
    # "ai21.j2-mid-v1"
    embeddings = BedrockEmbeddings(client=active_session, credentials_profile_name="bedrock-admin", model_id="ai21.j2-mid-v1")

    store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        distance_strategy=DistanceStrategy.EUCLIDEAN
    )

    retriever = store.as_retriever(search_kwargs={"k": 3})
    # retriever = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .7})

    def initialize_bedrock_and_qa(bedrock_session, retriever, chain_type_kwargs):
        llm = Bedrock(
            client=bedrock_session,
            credentials_profile_name="bedrock-admin",
            # model_id="amazon.titan-tg1-large",
            model_id="ai21.j2-mid-v1"
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            combine_docs_chain_kwargs=chain_type_kwargs,
            return_source_documents=True,
            condense_question_llm=llm
        )

        return llm, qa

    # Define or fetch other required parameters
    # retriever = store.as_retriever(search_kwargs={"k": 3})
    chain_type_kwargs = {"prompt": PROMPT}

    # Call the function and store the returned values in variables
    llm, qa = initialize_bedrock_and_qa(active_session, retriever, chain_type_kwargs)

    # ending section for load qa()

    app.mount("/static", StaticFiles(directory="static"), name="static")

    def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
        valid_username = USER_CREDENTIALS["username"]
        valid_password = USER_CREDENTIALS["password"]
        if not (credentials.username == valid_username and credentials.password == valid_password):
            raise HTTPException(status_code=401, detail="Invalid username or password")

    @app.get("/")
    def home():
        return JSONResponse(content={"message": "Welcome to the API!"})

    @app.get("/search/{project}")
    def search_docs(project: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
        return JSONResponse(content={"project": project})

    @app.get("/chat/{project}")
    def chat(project: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
        return JSONResponse(content={"project": project})

    def ask_question(qa, question, chat_history):
        result = qa({"question": question,"chat_history": chat_history})
        chat_history = chat_history.append((question, result["answer"]))
        print("Question:", question)
        print("Answer:", result["answer"])
        return result

    @app.get("/getChatBotResponse_project")
    def get_bot_response_project(
        msg: str, project: str, with_resp: bool = True, credentials: HTTPBasicCredentials = Depends(authenticate)
    ):
        # msg=input()
        ret = {}
        ret["query"] = msg
        ret["result"] = ""
        ret["Flag"] = None
        ret["num_of_documents"] = 0
        ret["document"] = {}

        docs = ask_question(qa, msg, chat_history)
        # print(docs)
        d = docs["source_documents"]

        repl = "NewCo"
        subs = "anthem"
        subs = ["anthem", "humana"]
        # join the list of substrings with a pipe separator to create a regex pattern
        pattern = "|".join(map(re.escape, subs))
        # regex used for ignoring cases and replacing all substrings
        res = re.sub("(?i)" + pattern, lambda m: repl, docs["answer"])
        ret["result"] = res

        if docs["answer"].strip().strip("\n") == "Could not find the relevant information. Please try again by rephrasing your question." or docs["answer"].strip().strip("\n") ==  "Could not find the relevant information. Please try again by rephrasing your question":
            ret["Flag"] = "True"
        else:
            ret["Flag"] = "False"

        if with_resp:
            context_list = []
            if ret['Flag'] == "False":
                for i, record in enumerate(d):
                    file_name = str(os.path.basename(record.metadata["source"]))
                    content = str(os.path.basename(record.page_content))
                    match = re.search(r'[.!?]', content)
                    if match:
                        sentence_end_index = match.end()
                    else:
                        sentence_end_index = 0
                    refined_content_op = content[sentence_end_index:].strip()
                    refined_content = ' '.join(re.split(r'(?<=[.:;])\s', refined_content_op)[:2])
                    # refined_content_op = refined_content.replace("\n",'')
                    refined_content = refined_content.replace("\n",'')
                    tmp_content = refined_content.replace("\t",'')
                    tmp_content = ''.join(e for e in refined_content if e.isalnum())
                    connection_string = os.getenv("CONNECTION_STRING")
                    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                    container_client = blob_service_client.get_container_client(container_name)
                    blob_client = container_client.get_blob_client(file_name)
                    blob_url = blob_client.url
                    temp_pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf").name
                    urllib.request.urlretrieve(blob_url, temp_pdf_path)
                    with pdfplumber.open(temp_pdf_path) as pdf:
                        for page in pdf.pages:
                            text = page.extract_text()
                            text = text.replace("\n",'')
                            text = text.replace("\t",'')
                            tmp_text = ''.join(e for e in text if e.isalnum())
                            if re.search(tmp_content,tmp_text):
                                encoded_page_number = urllib.parse.quote(str(page.page_number))
                                page_url = blob_url + "#page=" + encoded_page_number

                    try:
                        context_list.append({"url": page_url, "content": refined_content_op, })

                    except:
                        # page_number=1
                        encoded_page_number = urllib.parse.quote(str(1))
                        page_url = blob_url + "#page=" + encoded_page_number
                        context_list.append({"url": page_url, "content": refined_content_op,})

                ret["document"] = context_list
                ret["num_of_documents"] = len(d)

        return ret

    @app.get("/generate/{project}")
    def generate_embeddings(project: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
        from generate_embeddings_lc import create_vectors

        ret = create_vectors(project)

        final_msg = f"Embeddings created for {project}<br>"
        final_msg += ret
        return final_msg

    @app.get("/evaluate/{project}")
    def evaluate(project: str, credentials: HTTPBasicCredentials = Depends(authenticate)):
        from langchain.evaluation.qa import QAEvalChain

        final_msg = ""

        module_name = f"data.{project}.evaluate.questions"
        module = import_module(module_name)
        examples = module.examples

        predictions = qa.apply(examples)

        eval_chain = QAEvalChain.from_llm(llm)
        graded_outputs = eval_chain.evaluate(examples, predictions)

        for i, eg in enumerate(examples):
            final_msg += f"Example {i}:<br>"
            final_msg += "Question: " + predictions[i]["query"] + "<br>"
            final_msg += "Real Answer: " + predictions[i]["answer"] + "<br>"
            final_msg += "Predicted Answer: " + predictions[i]["result"] + "<br>"
            final_msg += "Predicted Grade: " + graded_outputs[i]["text"] + "<br><br>"

        return final_msg

    @app.get("/getAESummary")
    def get_bot_response_project(
            msg: str, temp: float, top_p:float, maxCount:int, with_resp: bool = True, credentials: HTTPBasicCredentials = Depends(authenticate)):
        llm2 = Bedrock(client=active_session, credentials_profile_name="bedrock-admin", model_id="ai21.j2-mid-v1")
        llm2.model_kwargs = {'temperature':temp, 'topP':top_p, 'maxTokens':maxCount} # 'max_tokens_to_sample'
        result = llm2.predict(msg)
        result = re.sub(r'\n',"",result, 1)
        print("Message: "+msg)
        print("Result: "+result)

        return result

    return app


async def stop_server_after_delay(server, delay):
    await asyncio.sleep(delay)
    print("Stopping the server after delay...")
    server.should_exit = True

if __name__ == "__main__":
    while True:
        app = run_main_script()

        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = uvicorn.Server(config=config)
        loop = asyncio.get_event_loop()

        # Set the delay in seconds after which the server should be stopped
        stop_delay = 7190  # Example: Stop after 2 hrs - 10 secs

        # Create a task to stop the server after the delay
        loop.create_task(stop_server_after_delay(server, stop_delay))

        # Start the server asynchronously
        loop.run_until_complete(server.serve())

        print("Server has been stopped.")


if __name__ == "__main__":

    uvicorn.run("main:app", host="127.0.0.1", port=8000)
