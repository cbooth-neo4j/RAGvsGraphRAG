import datetime
import logging
import sys
from typing import Sequence, Type, List

import yaml
from dotenv import load_dotenv
import os,subprocess

from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from langchain_core.utils.json_schema import dereference_refs

import google
from google.genai.types import ThinkingConfig, HttpOptions
from google.genai import types
from google.auth.transport import Request
from google.oauth2.credentials import Credentials
from google.cloud.aiplatform_v1.services.prediction_service import PredictionServiceClient
from google.cloud.aiplatform_v1.services.prediction_service.transports import PredictionServiceRestTransport

from pydantic import BaseModel

from genai_common.core.init_vertexai import init_vertexai

from genai_common.proxy_token_roller import helix_proxy_token_roller, coin_proxy_token_roller
from genai_common.config.environment import R2D2Environment, VertexAIEnvironment


load_dotenv()
os.environ['SSL_CERT_FILE'] = os.getenv('R2D2_CERT_PATH')
from utils.graph_rag_logger import  setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Load LLM

#token_roller = helix_proxy_token_roller()

r2d2 = R2D2Environment()
print(r2d2)
vertex_env = VertexAIEnvironment()
coin_credentials = yaml.safe_load(r2d2.coin_consumer_credentials_path.read_text())

token_roller = coin_proxy_token_roller(
    url=r2d2.coin_consumer_endpoint_url,
    client_id=coin_credentials["client_id"],
    client_secret=coin_credentials["client_secret"],
    scope=r2d2.coin_consumer_scope,
    # ssl_cert_file= './certs/cacerts.pem' #r2d2.ssl_cert_file,
)



print('Loaded all modules in utils.llm', flush=True)
logger.debug('Loaded all modules in utils.llm...')

def  get_coin_token():

    command = "helix auth access-token print -a"
      #Execute the command and capture its output
    result = subprocess.check_output(command, shell = True)
    return result.decode().strip()

# cert from https://cedt-confluence.nam.nsroot.net/confluence/download/attachments/816654701/CitiInternalCAChain_PROD.pem (has to be a PROD cert)

headers={}

def generate_dereferenced_schema(model: Type[BaseModel]) -> dict:
    """
    Returns a JSON schema for the given Pydantic model where references
    like `#/defs/Name` are fully dereferenced (inlined).
    This is required for Gemini VertexAI to work because their protobuf based schema
    doesn't support nested references. The returned json objects can be converted
    back into the original pydantic objects because they will have identical json
    structure.
    """
    raw_schema = model.model_json_schema(ref_template="#/defs/{model}")
    if "$defs" in raw_schema:
        raw_schema["defs"] = raw_schema.pop("$defs")

    inlined = dereference_refs(raw_schema)
    inlined.pop("defs", None)
    return inlined



# def get_new_token(request: Request, scopes: Sequence[str]) -> tuple[str, datetime.datetime]:
#     """
#     Refreshes the token and returns the new token and the time of refresh.
#     """
#     # Vertex library will call this function 3:45 before the token expires and throw an exception if it doesn't get
#     # a new one.
#     token, expiry = token_roller.get_token_and_expiry(flush_cache=True)
#     return token, expiry.replace(tzinfo=None)

def get_new_token(request: Request, scopes: Sequence[str])  -> tuple[str, datetime.datetime]:
    """
    Refreshes the token and returns the new token and the time of refresh.
    """
    # Vertex library will call this function 3:45 before the token expires and throw an exception if it doesn't get
    # a new one.

    coin_credentials = yaml.safe_load(r2d2.coin_consumer_credentials_path.read_text())

    token_roller = coin_proxy_token_roller(
        url=r2d2.coin_consumer_endpoint_url,
        client_id=coin_credentials["client_id"],
        client_secret=coin_credentials["client_secret"],
        scope=r2d2.coin_consumer_scope,
        #ssl_cert_file= './certs/cacerts.pem' #r2d2.ssl_cert_file,
    )

    token, expiry = token_roller.get_token_and_expiry(flush_cache=True)
    return token, expiry.replace(tzinfo=None)

def get_google_client():
    VERTEX_CREDENTIALS = Credentials(token=None, refresh_handler=get_new_token)
    client = google.genai.Client(credentials=VERTEX_CREDENTIALS,
                          vertexai=True,
                          project=vertex_env.vertexai_project,
                          location="us-central1",
                          http_options=types.HttpOptions(base_url=vertex_env.vertexai_api_endpoint))
    return client

def def_count_tokens_vertex(text: str):
    client = get_google_client()
    response = client.models.count_tokens(
        model="gemini-2.5-flash",
        contents=text,
    )
    return response

def get_vertex_llm(schema = None, response_type="text/plain", system_prompt: str=""):
    logger.debug("Returning Vertex LLM....")
    VERTEX_CREDENTIALS = Credentials(token=None, refresh_handler=get_new_token)
    init_vertexai(vertex_env, token_roller)
    #client = genai.Client(http_options=HttpOptions(base_url='https://r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex',                     apiVersion='v1'))

    #vertexai_config = VertexAIConfig(credentials=VERTEX_CREDENTIALS,   project="prj-gen-ai-9571",  location="us-central1")

    generate_content_config = types.GenerateContentConfig(
        temperature = float(os.getenv('MODEL_TEMPERATURE', '0.0')),
        top_p = 1,
        seed = int(os.getenv('MODEL_SEED', '42')) if os.getenv('MODEL_SEED') else 42,
        max_output_tokens = int(os.getenv('MAX_TOKENS')) if os.getenv('MAX_TOKENS') else None,
        response_mime_type = response_type,
        response_schema=schema,
        thinking_config=ThinkingConfig(includeThoughts=False),
        system_instruction=[
            types.Part.from_text(
                text=system_prompt)],
                                                 #You are an experienced Payments Agent, Plutus, at Citi's Treasury and Trade Services (TTS) responsible for answering all questions around Payments."
    )
    model_kwargs = {
        "vertex_config" : {
            "temperature" : 0.0,
            "top_p": 1,
            "seed":  0,
            "max_output_tokens": 65535,
            "response_mime_type": response_type,
            "response_schema": schema,
            "thinking_config":  ThinkingConfig(includeThoughts=False),
            "system_instruction": [types.Part.from_text(text=system_prompt)]
        }
    }
    rest_transport = PredictionServiceRestTransport(
        credentials=VERTEX_CREDENTIALS,
        host="r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex"
    )
    rest_pred_client = PredictionServiceClient(transport=rest_transport)
    llm = ChatVertexAI(temperature = 0.0,
                        top_p = 1,
                        seed = 0,
                        max_output_tokens = 65535,
                       project="prj-gen-ai-9571",
                       location="us-central1",
                       credentials=VERTEX_CREDENTIALS,
                       endpoint_version="v1",
                       model="gemini-2.5-pro",
                       response_schema=schema,
                       include_thoughts=False,
                       response_mime_type=response_type,
                       client=rest_pred_client,
                       api_endpoint=vertex_env.vertexai_api_endpoint,
                       model_kwargs=model_kwargs,
                       max_retries=10)

    llm.client._prediction_client = rest_pred_client
    return llm


def get_vertex_embeddings(model_name: str = "text-embedding-005", project_name: str = "prj-gen-ai-9571", location: str = "us-central1") -> VertexAIEmbeddings:
    init_vertexai(vertex_env, token_roller)
    VERTEX_CREDENTIALS = Credentials(token=None, refresh_handler=get_new_token)
    rest_transport = PredictionServiceRestTransport(
        credentials=VERTEX_CREDENTIALS,
        host="r2d2-c3p0-icg-msst-genaihub-178909.apps.namicg39023u.ecs.dyn.nsroot.net/vertex"
    )
    rest_pred_client = PredictionServiceClient(transport=rest_transport)
    embeddings = VertexAIEmbeddings(model_name=model_name,
                                    client=rest_pred_client,
                                    credentials=VERTEX_CREDENTIALS,
                                    api_endpoint=vertex_env.vertexai_api_endpoint,
                                    project= project_name,
                                    location=location)
    return embeddings

from ragas.llms.base import LangchainLLMWrapper

class SyncOnlyWrapper(LangchainLLMWrapper):
    async def agenerate_text(self, *args, **kwargs):
        return self.generate_text(*args, **kwargs)