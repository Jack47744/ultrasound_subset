# Use an official Python runtime as a parent image
FROM python:3.10.12-slim-buster

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the local directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME=World

# Copy the entrypoint script into the container and make it executable
COPY entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh

# Run entrypoint.sh when the container launches
ENTRYPOINT ["entrypoint.sh"]
