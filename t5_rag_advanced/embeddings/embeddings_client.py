# import json
# import requests
from sentence_transformers import SentenceTransformer


class EmbeddingsClient:
    # _endpoint: str
    # _api_key: str

    def __init__(self, endpoint: str, model_name: str, api_key: str):
        # if not api_key or api_key.strip() == "":
        #     raise ValueError("API key cannot be null or empty")
        # self._endpoint = endpoint
        # self._api_key = "Bearer " + api_key
        # self._model_name = model_name
        self._local_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(
            self, inputs: str | list[str],
            dimensions: int,
            print_response: bool = False
    ) -> dict[int, list[float]]:
        if isinstance(inputs, str):
            inputs = [inputs]

        vectors = self._local_model.encode(inputs, normalize_embeddings=True)
        return {i: v.tolist() for i, v in enumerate(vectors)}

        # --- OpenAI implementation (commented out) ---
        # headers = {
        #     "Content-Type": "application/json",
        #     "Authorization": self._api_key
        # }
        # data = {
        #     "input": inputs,
        #     "model": self._model_name,
        #     "dimensions": dimensions
        # }
        # response = requests.post(self._endpoint, headers=headers, json=data)
        # if response.status_code != 200:
        #     raise Exception(f"OpenAI API error: {response.text}")
        # response_json = response.json()
        # if print_response:
        #     print(json.dumps(response_json, indent=2))
        # embeddings_dict = {}
        # for item in response_json["data"]:
        #     embeddings_dict[item["index"]] = item["embedding"]
        # return embeddings_dict
