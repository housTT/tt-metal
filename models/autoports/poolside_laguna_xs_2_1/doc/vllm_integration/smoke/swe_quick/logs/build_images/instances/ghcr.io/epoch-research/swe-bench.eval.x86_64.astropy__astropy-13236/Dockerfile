FROM --platform=linux/x86_64 ghcr.io/epoch-research/swe-bench.env.x86_64.6fc76dab062eec6ac871dd

COPY ./setup_repo.sh /root/
RUN /bin/bash /root/setup_repo.sh

WORKDIR /testbed/
