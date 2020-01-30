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
RUN cd /ci && ./build_thirdparty.sh && rm -r /ci

USER conan
WORKDIR /app
CMD [ "/bin/bash" ]
