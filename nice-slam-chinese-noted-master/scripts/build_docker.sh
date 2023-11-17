REMOTE_NAME="immortalqx"
DOCKER_NAME="nice-slam"
DOCKER_TAG="1.0"

echo $REMOTE_NAME/$DOCKER_NAME:$DOCKER_TAG

docker build . \
	-f Dockerfile \
	-t $REMOTE_NAME/$DOCKER_NAME:$DOCKER_TAG \
	--network host \
	--build-arg HTTP_PROXY=http://127.0.0.1:7890 \
	--build-arg HTTPS_PROXY=http://127.0.0.1:7890
