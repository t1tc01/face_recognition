# face_recognition
## Run docker triton inference server
docker run -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:23.06-py3

tritonserver --model-repository=/models


