# User image Dockerfile template
# image_key: instance_image_key corresponding to instance_id
# apt_packages: split by ' ', load multi packages needed to be installed in ubuntu
_DOCKERFILE_USER_IMAGE_PY = r"""
FROM {image_key}

RUN apt-get update && apt-get install -y --no-install-recommends {apt_packages}
"""
