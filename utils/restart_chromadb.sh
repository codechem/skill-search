rm -rf ./chroma-data/*
docker stop chromadb
docker rm chromadb
docker run --name chromadb -v ./chroma-data:/data -p 8000:8000 -d chromadb/chroma 