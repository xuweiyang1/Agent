# A minimal image for the filesystem MCP server and its FastAPI surface.
#
# The point of this file is not to be clever; it is to make the deployment
# step in the ROADMAP concrete and reproducible. It packages exactly the
# runtime dependencies and the one ASGI app that already exists
# (``mcp_server/cloud.py``), and nothing else.
#
# python:3.10-slim matches the interpreter the project is developed and tested
# on. That is deliberate: W4's static ``interrupt_before`` breakpoint exists
# because 3.10 cannot use ``interrupt()`` under async, so a 3.11 base would
# ship a runtime the test suite never proved.
FROM python:3.10-slim

WORKDIR /app

# requirements.txt first, on its own layer. Dependencies change far less often
# than the source, so this keeps a rebuild after a code edit from re-resolving
# every wheel.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Unbuffered output, or a container log lags by a buffer and cloud debugging
# gets harder than it needs to be.
ENV PYTHONUNBUFFERED=1

# The cloud host sets PORT; cloud.py reads it and defaults to 8080, which is
# the port this image advertises.
EXPOSE 8080

# Host 0.0.0.0, not the library default 127.0.0.1: inside a container the
# loopback is not reachable from the host's port mapping, so binding localhost
# yields a container that looks healthy and answers nothing.
CMD ["python", "-m", "mcp_server.cloud", "--host", "0.0.0.0"]