github-login:
	echo $(GITHUB_MILABENCH_TOK) | sudo docker login ghcr.io -u Delaunay --password-stdin


docker-ngc: docker-ngc-build docker-ngc-push

docker-ngc-build:
	sudo docker build 														\
		--build-arg CACHEBUST=`git rev-parse $(git branch --show-current)`	\
	 	-f docker/Dockerfile-ngc 											\
		-t milabench:cuda-ngc-25.10 . 

	sudo docker tag 														\
		milabench:cuda-ngc-25.10											\
	 	ghcr.io/mila-iqia/milabench:cuda-ngc-25.10

docker-ngc-push:
	sudo docker push ghcr.io/mila-iqia/milabench:cuda-ngc-25.10

docker-ngc-run:
	sudo docker run	-it --rm --ipc=host --network=host            	\
		--runtime=nvidia --gpus all                          	\
		--security-opt=label=disable                          	\
		-e HF_TOKEN=$HF_TOKEN									\
		-e MILABENCH_HF_TOKEN=$HF_TOKEN									\
		-v "$MILABENCH_BASE/runs:/milabench/results/runs"			\
		-v "$MILABENCH_BASE/data:/milabench/results/data" 			\
		ghcr.io/mila-iqia/milabench:cuda-ngc-25.10 bash -c "(. /milabench/env/bin/activate && milabench run --use-current-env)"


docker-ngc-edit:
	sudo docker run	-it --ipc=host --network=host            	\
		--runtime=nvidia --gpus all                          	\
		--security-opt=label=disable                          	\
		-e HF_TOKEN=$HF_TOKEN									\
		-e MILABENCH_HF_TOKEN=$HF_TOKEN									\
		-v "$MILABENCH_BASE/runs:/milabench/results/runs"			\
		-v "$MILABENCH_BASE/data:/milabench/results/data" 			\
		ghcr.io/mila-iqia/milabench:cuda-ngc-25.10 bash 
