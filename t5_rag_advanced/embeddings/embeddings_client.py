import json

import requests




class EmbeddingsClient:
    _endpoint: str
    _api_key: str

    def __init__(self, endpoint: str, model_name: str, api_key: str):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be null or empty")

        self._endpoint = endpoint
        self._api_key = "Bearer " + api_key
        self._model_name = model_name

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int,
            print_response: bool = False
    ) -> dict[int, list[float]]:
        """
        Generate dict of indexed embeddings:
            inputs[0](text) -> [0][embedding]
            inputs[1](text) -> [1][embedding]
            ...

        Args:
            inputs: input text, can be singular string or list of strings
            dimensions: number of dimensions
            print_response: to print response in chat or not
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": self._api_key
        }

        data = {
            "input": inputs,
            "model": self._model_name,
            "dimensions": dimensions
        }

        response = requests.post(self._endpoint, headers=headers, json=data)

        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.text}")

        response_json = response.json()
        if print_response:
            print(json.dumps(response_json, indent=2))

        embeddings_dict = {}
        for item in response_json["data"]:
            embeddings_dict[item["index"]] = item["embedding"]

        return embeddings_dict


# Hint:
# Request:
# curl https://api.openai.com/v1/embeddings \
#   -H "Content-Type: application/json" \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -d '{
#     "input": "Your text string goes here",
#     "model": "text-embedding-3-small",
#     "dimensions": 384
#   }'
#
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }