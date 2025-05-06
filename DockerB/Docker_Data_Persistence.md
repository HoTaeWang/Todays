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

