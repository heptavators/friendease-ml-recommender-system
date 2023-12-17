# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the entire project into the container at /app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5050 available to the world outside this container
EXPOSE 5050

# Run uvicorn when the container launches
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "5050", "--lifespan", "on", "--loop", "asyncio"]
