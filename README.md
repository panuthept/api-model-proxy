# api-model-proxy


```python
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

client = OpenAI()
proxy = LoggingProxy(client)
proxy.deploy(host="localhost", port=8000)
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000")
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response.choices[0].message.content)
```