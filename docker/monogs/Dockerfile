FROM duag-slam-base:latest
COPY repos/MonoGS /opt/MonoGS
RUN cd /opt/MonoGS && \
    pip install -e submodules/diff-gaussian-rasterization && \
    pip install -e submodules/simple-knn
