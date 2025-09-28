FROM opencfd/openfoam-default:latest

# Set up environment
ENV PATH="/opt/openfoam/bin:${PATH}"
ENV FOAM_TUTORIALS=/opt/openfoam/tutorials
ENV WM_PROJECT_DIR=/opt/openfoam

# Copy workspace contents
COPY . /home/openfoam/lungCase/

# Set working directory
WORKDIR /home/openfoam/lungCase

# Source OpenFOAM environment
RUN echo '. /opt/openfoam/etc/bashrc' >> /etc/profile && \
    echo '. /opt/openfoam/etc/bashrc' >> ~/.bashrc