FROM duag-slam-base:latest
COPY repos/dpgo /opt/dpgo
RUN mkdir -p /opt/dpgo/build && \
    cd /opt/dpgo/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)
COPY extracted/dpgo_wrapper /opt/dpgo_wrapper
RUN cd /opt/dpgo_wrapper && python setup.py build_ext --inplace
