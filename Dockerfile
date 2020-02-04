FROM conanio/gcc8

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root
RUN export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y wget libtbb-dev libglew-dev qt5-default libxkbcommon-dev libflann-dev libpng-dev libgtkglext1-dev && apt clean
RUN export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y libvtk7-dev && apt clean
RUN export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y libboost-all-dev && apt clean
RUN export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra && apt clean
RUN export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y software-properties-common && apt clean
RUN export DEBIAN_FRONTEND=noninteractive && add-apt-repository -y ppa:deadsnakes/ppa
RUN export DEBIAN_FRONTEND=noninteractive && apt update && apt install -y python3.7 && apt clean

ADD ci /ci
RUN cd /ci && ./build_thirdparty.sh
RUN cd /ci && ./copy_thirdparty.sh

COPY generateVirtualSequence /ci/tmp/generateVirtualSequence/
COPY build_generateVirtualSequence.sh /ci/tmp/
RUN cd /ci/tmp && ./build_generateVirtualSequence.sh

COPY matchinglib_poselib /ci/tmp/matchinglib_poselib/
COPY build_matchinglib_poselib.sh /ci/tmp/
RUN cd /ci/tmp && ./build_matchinglib_poselib.sh

WORKDIR /app
RUN cp -r /ci/tmp/thirdparty /app/
RUN cp -r /ci/tmp/tmp/ /app/
#RUN rm -r /ci

USER conan
RUN echo 'alias python=python3' >> ~/.bashrc
CMD [ "/bin/bash" ]
