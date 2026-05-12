from sentence_transformers import SentenceTransformer


class EmbeddingsClient:

    def __init__(self, endpoint: str, model_name: str, api_key: str):
        self._model = SentenceTransformer(model_name)

    def get_embeddings(
            self,
            inputs: str | list[str],
            dimensions: int,
            print_response: bool = False,
    ) -> dict[int, list[float]]:
        """
        Generate dict of indexed embeddings:
            inputs[0](text) -> [0][embedding]
            inputs[1](text) -> [1][embedding]
            ...
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        vectors = self._model.encode(inputs, normalize_embeddings=True)
        return {i: v.tolist() for i, v in enumerate(vectors)}


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