# ScholarScope node image. The SAME image runs on all 3 EC2 nodes; ROLE=master|worker picks what the entry script does.
FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        g++ openmpi-bin libopenmpi-dev zlib1g-dev openssh-server openssh-client python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root MPI user (OpenMPI refuses to run as root). uid 1000 = the 'ubuntu' user on EC2, so mounted files keep one owner.
# Password '*' = no password can ever match, but key login still works (a locked '!' account would be refused by sshd).
RUN useradd -m -u 1000 -s /bin/bash mpiuser && usermod -p '*' mpiuser && mkdir -p /data && chown mpiuser:mpiuser /data

# sshd for MPI: port 2222 (the host's own sshd keeps 22), key login only, only mpiuser.
RUN mkdir -p /run/sshd && ssh-keygen -A && printf '%s\n' \
        'Port 2222' 'PubkeyAuthentication yes' 'PasswordAuthentication no' 'KbdInteractiveAuthentication no' \
        'PermitRootLogin no' 'AllowUsers mpiuser' > /etc/ssh/sshd_config.d/mpi.conf
# ssh client config for mpiuser: Port 2222, no host-key questions. System-wide, because ~/.ssh is the mounted key folder.
RUN printf '%s\n' 'Host *' '    Port 2222' '    StrictHostKeyChecking no' '    UserKnownHostsFile /dev/null' '    LogLevel ERROR' \
        > /etc/ssh/ssh_config.d/mpi.conf

# The engine, compiled exactly as the PDF says (+ zlib for the .gz shards).
COPY engine/ /src/engine/
RUN mpicxx -O3 -fopenmp -std=c++17 /src/engine/engine.cpp -o /opt/engine -lz && chmod 755 /opt/engine

# The provided server, unchanged (the grader checks its sha256).
COPY server.py /opt/server.py

COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
