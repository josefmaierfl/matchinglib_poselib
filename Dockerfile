FROM conanio/gcc8

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root
#RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libvtk7-dev && apt-get clean
#RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libboost-all-dev && apt-get clean
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
	gfortran && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && add-apt-repository -y ppa:deadsnakes/ppa
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y python3.6 && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libglu1-mesa-dev mesa-common-dev mesa-utils freeglut3-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y qt5-default && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libgoogle-glog-dev libatlas-base-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libsuitesparse-dev && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y nano && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libomp-dev ccache && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y libhdf5-dev libhdf5-serial-dev ccache && apt-get clean
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update && apt-get install -y curl && apt-get clean

ADD ci /ci
RUN cd /ci && ./build_thirdparty.sh
RUN cd /ci && ./copy_thirdparty.sh
RUN python3 -m pip install --upgrade pip setuptools wheel
ADD py_test_scripts/requirements.txt .
RUN python3 -m pip install -r requirements.txt && rm requirements.txt
#RUN python3 -m pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 python3 -m pip install -U

COPY generateVirtualSequence /ci/tmp/generateVirtualSequence/
COPY build_generateVirtualSequence.sh /ci/tmp/
RUN cd /ci/tmp && ./build_generateVirtualSequence.sh

COPY matchinglib_poselib /ci/tmp/matchinglib_poselib/
COPY build_matchinglib_poselib.sh /ci/tmp/
RUN cd /ci/tmp && ./build_matchinglib_poselib.sh

WORKDIR /app
RUN cp -r /ci/tmp/thirdparty /app/
RUN cp -r /ci/tmp/tmp/. /app/
COPY start_testing.sh /app/
#RUN rm -r /ci

RUN chown -R conan /app

USER conan
RUN echo 'alias python=python3' >> ~/.bashrc
CMD [ "/bin/bash" ]
