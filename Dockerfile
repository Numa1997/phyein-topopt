# phyein-topopt toolchain: one pinned image for this session, GitHub runners, Kaggle and a PC.
# Base pinned by digest so a rebuild months later gives the same system libraries.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

# ffmpeg: video; libgl/libegl/libxrender/libosmesa: VTK off-screen rendering without a display
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libegl1 libxrender1 libosmesa6 libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/phyein-topopt/requirements.txt
# optional build secret "ca": a CA bundle for builds behind a TLS-inspecting proxy
# (docker build --secret id=ca,src=ca.crt .). Used only during this step, never stored in the image.
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install --no-cache-dir -r /opt/phyein-topopt/requirements.txt

COPY topopt /opt/phyein-topopt/topopt
COPY examples /opt/phyein-topopt/examples
COPY benchmarks /opt/phyein-topopt/benchmarks
ENV PYTHONPATH=/opt/phyein-topopt PYVISTA_OFF_SCREEN=true MPLBACKEND=Agg
WORKDIR /work

# default: run the acceptance checks (2D FE vs beam theory, 3D element/cantilever/gradient)
CMD ["sh", "-c", "python /opt/phyein-topopt/benchmarks/cantilever2d.py && python /opt/phyein-topopt/benchmarks/check3d.py"]
