FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the benchmark script into the container
COPY benchmark.py .

# Define the command to run when the container starts
CMD ["python", "benchmark.py"]

# Set a volume for storing the results outside the container
VOLUME /app/results