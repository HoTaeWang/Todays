# Practical Docker 



## How to check docker installation correct

```
docker run --rm hello-world
```

```
docker run -it ubuntu bash
```



## Docker 관련 전문 용어

* Layers
* Docker Image
* Docker Container
  * A Docker image, when it's run in a host computer, spawns a process with it own namespace, known as a Docker container.
* Bind Mounts and Volumes
* Docker Registry
  * Docker Hub
  * Google Container Registry
  * Amazon Elastic Container Registry
  * JFrog Artifactory
* Dockerfile
  * FROM
  * ENV
  * RUN
  * CMD
  * ENTRYPOINT
* Docker Engine
  * Docker daemon
  * Docker CLI
    * docker build
    * docker pull
    * docker run
    * docker exec
    * docker help
  * Docker API
    * Docker also provides an API for interacting with Docker Engine. The simplest way to get started by Docker API is to use curl to send an API request. For Windows Docker hosts, we can reach the TCP endpoint:

```
curl http://localhost:2375/images/json
[{"Containers":-1,"Created":1511223798,"Id":"sha256:f2a91732366c0332ccd7afd2a5c4ff2b9af81f549370f7a19acd460f87686bc7","Labels":null,"ParentId":"","RepoDigests":
["hello-
world@sha256:66ef312bbac49c39a89aa9bcc3cb4f3c9e7de3788c944158df3ee0176d32b751"],"RepoTags":
["hello-
world:latest"],"SharedSize":-1,"Size":1848,"VirtualSize":1848}
```

* Docker compose
* Docker info

```
docker info
```

* Working with Docker Images

```
docker image ls
docker images
```

* docker inspect
  Of importance are the image properties "Env", "Cmd", and "Layers"

```
docker image inspect hello-world
```



## To start a container and run the associated image

```
docker run -p 80:80 nginx
```

```
docker run -p 8080:80 nginx
curl http://localhost:8080
```



## Docker Process

```
docker ps
docker stop <container-id>
docker kill <container-id>
docker ps -a
docker rm <container-id>
docker image ls
docker rmi <image-id>
```

