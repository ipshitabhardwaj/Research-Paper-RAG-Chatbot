from utils.embeddings import EmbeddingModel

embedding = EmbeddingModel()

model = embedding.get_embedding_model()

text = "What is Retrieval Augmented Generation?"

vector = model.embed_query(text)

print("="*60)
print("Embedding Generated Successfully")
print("="*60)

print(f"\nDimension : {len(vector)}")

print("\nFirst 10 Values\n")

print(vector[:10])