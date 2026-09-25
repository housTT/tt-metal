FROM --platform=linux/x86_64 ghcr.io/epoch-research/swe-bench.base.x86_64
RUN git clone -o origin https://github.com/astropy/astropy /testbed \
    && chmod -R 777 /testbed  # so nonroot user can run tests
COPY ./setup_env.sh /root/
RUN chmod +x /root/setup_env.sh
RUN /bin/bash -c "source ~/.bashrc && /root/setup_env.sh"

WORKDIR /testbed/

# Automatically activate the testbed environment
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate testbed" > /root/.bashrc
