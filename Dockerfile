FROM conanio/gcc11

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y software-properties-common
#RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y apt-utils
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y wget build-essential sudo cmake pkg-config
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libboost-all-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libtbb2 libtbb-dev libpng-dev libtiff-dev libjpeg-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libshine-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y qt5-default 
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libflann-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libgtk-3-dev 
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev openexr libhdf5-dev libomp-dev && apt-get clean

#For Ceres solver
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libgoogle-glog-dev libgflags-dev
#RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libatlas-base-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libopenblas-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y liblapack-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y libsuitesparse-dev
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y ocl-icd-opencl-dev && apt-get clean

ADD ci /ci
RUN cd /ci && sed -i 's/\r$//' build_thirdparty.sh  && \  
    chmod +x build_thirdparty.sh
RUN cd /ci && sed -i 's/\r$//' build_eigen.sh  && \  
    chmod +x build_eigen.sh
RUN cd /ci && sed -i 's/\r$//' make_opencv.sh  && \  
    chmod +x make_opencv.sh
RUN cd /ci && sed -i 's/\r$//' build_ceres.sh  && \  
    chmod +x build_ceres.sh
RUN cd /ci && ./build_thirdparty.sh

# COPY matchinglib_poselib /ci/tmp/matchinglib_poselib/
# COPY build_matchinglib_poselib.sh /ci/tmp/
# RUN cd /ci/tmp && sed -i 's/\r$//' build_matchinglib_poselib.sh  && \  
#     chmod +x build_matchinglib_poselib.sh
# RUN cd /ci/tmp && ./build_matchinglib_poselib.sh

WORKDIR /app
# RUN cp -r /ci/tmp/tmp/. /app/
# COPY start_matchinglib_poselib.sh /app/
# RUN cd /app && sed -i 's/\r$//' start_matchinglib_poselib.sh  && \  
#     chmod +x start_matchinglib_poselib.sh

RUN chown -R conan /app

USER conan
CMD [ "/bin/bash" ]
