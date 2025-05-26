This is the backend of the personality test. It connects a GPU enabled API with a neural-network that matches word-structure to personality scores.

To launch the frontend ;
docker pull thales884/personaliytest 


docker run thales884/personaliytest -e BACKEND_GPU_ENABLED_API=<IP ADDRESS>:<PORT>

If you remove the character_embeddings_bigfive.json the script will recalculate the embedding vectors.
the scraper script can scrape Wikipedia for inforamtion an dactions on a given characther.
