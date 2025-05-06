# Data Persistence

* Containers were meant and designed for stateless workloads and the design of the container layers shows that.
  * The data is locked tightly to the host and makes running applications that share data across multiple containers and applications difficult
  * The data doesn't persist when the container is terminated and extracting the data out of the container is difficult.
  * Writing to the container's write layer requires a storage driver to manage the filesystem.
* Docker offers various strategies to persist the data
  * tmpfs mounts
  * Bind mounts
  * Volumes



## *tmpfs* Mounts

* tmpfs mounts are limited to Docker containers Only on linux 

```
# docker run -it --name tmpfs-test --mount type=tmpfs, target=/tmpfs-mount ubuntu bash
# docker run -it --name tmpfs-test --tmpfs /tmpfs-mount ubuntu bash
How to check
# docker inspect tmpsf-test
```



## *Bind* Mounts

* Only available only to Linux hosts
* In bind mounts, the file/directory on the host machine is mounted into the container.
* Example to try to mount our Docker host's home directory to a directory called '*host-home*' within the container.

```
# docker run -it --name mount-test --mount type=bind, source="$HOME", target=/host-home ubuntu bash
# docker run -it --name mount-test -v $HOME:/host-home ubuntu bash
How to check
# docker inspect mount-test
```

* bind can go rogue or a mistaken rm-rf can bring down the Docker host completely. To mitigate this, we can create a bind mount with the read-only option

```
# docker run -it --name mount-test --mount type=bind, source="$HOME", target=/host-home, readonly ubuntu bash
# docker run -it --name mount-test -v $HOME:/host-home:ro ubuntu bash
```



## Volumes

* Volumes are easier to back up or transfer than bind mounts
* Volumes work on both Linux and Windows containers
* Volumes can be shared among multiple containers without problems.



### Docker Volume Subcommands

* docker volume create
* docker volume inspect
* docker volume ls
* docker volume prune
* docker volume rm



### Create Volume

```
# docker volume create --name=<name of the volume> --label=<any extra metadata>
Example
# docker volume create --name=nginx-volume
```



### Inspect

* when you inspect any volume, the '*mountpoint*' property lists the location on the Docker host where the file containing the data of the volume is saved.

```
# docker volume inspect <name of the volume>
Example
# docker volume inspect nginx-volume
```



### List Volumes

* The list volume command shows all the volumes present on the host.

```
# docker volume ls
```

* prune volumes

  * The ***prune volume*** command removes all unused local volumes.

  ```
  # docker volume prune <--force>
  ```



### Remove volumes

* Even if the container stops, Docker will consider the volume to be in use
* Docker will not remove a volume that is in use and will return an error

```
# docker volume rm <name>
Example
# docker volume rm nginx-volume
```



### Using Volumes when starting a container

```
# docker run -it --name volume-test --mount target=/data-volume ubuntu bash
# docker run -it --name volume-test -v:/data-volume
Inspect
docker inspect volume-test
```

