FROM conanio/gcc8

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libvtk7-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libboost-all-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y software-properties-common apt-utils && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && add-apt-repository -y 'deb http://security.ubuntu.com/ubuntu xenial-security main'
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y build-essential cmake pkg-config && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y wget \
  libtbb2 \
	libtbb-dev \
	libglew-dev \
	qt5-default \
	libxkbcommon-dev \
	libflann-dev \
	libpng-dev \
	libgtk-3-dev \
	libgtkglext1 \
	libgtkglext1-dev \
	libtiff-dev \
	libtiff5-dev \
	libtiffxx5 \
	libjpeg-dev \
	libjasper1 \
	libjasper-dev \
	libavcodec-dev \
	libavformat-dev \
	libswscale-dev \
	libv4l-dev \
	libxvidcore-dev \
	libx264-dev \
	libdc1394-22-dev \
	openexr \
	libatlas-base-dev \
	gfortran \
  libhdf5-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libglu1-mesa-dev mesa-common-dev mesa-utils freeglut3-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libomp-dev ccache && apt-get clean

ADD ci /ci
RUN cd /ci && ./build_thirdparty.sh
#RUN cd /ci && ./copy_thirdparty.sh

COPY matchinglib_poselib /ci/tmp/matchinglib_poselib/
COPY build_matchinglib_poselib.sh /ci/tmp/
RUN cd /ci/tmp && ./build_matchinglib_poselib.sh

WORKDIR /app
#RUN cp -r /ci/tmp/thirdparty /app/
RUN cp -r /ci/tmp/tmp/. /app/
COPY start_matchinglib_poselib.sh /app/
#RUN rm -r /ci

RUN chown -R conan /app

USER conan
CMD [ "/bin/bash" ]
