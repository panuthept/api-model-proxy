from openai import OpenAI
from api_model_proxy import APIModelProxy

class LoggingProxy(APIModelProxy):
    def __init__(self, openai_client: OpenAI):
        super().__init__(openai_client)

    def _preprocess_request(self, request):
        print(f"Request: {request}")
        return request

    def _postprocess_response(self, response):
        print(f"Response: {response}")
        return response

client = OpenAI(
    api_key="sk-ismMdKm5GkbkX-g6Apx1HQ",
    base_url="https://api.sea-lion.ai/v1",
)
proxy = LoggingProxy(client)
proxy.deploy(host="localhost", port=8000)