# use a lightweight python image
FROM python:3.9-slim

# set the working directory in the container
WORKDIR /app

# copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all our files (app.py, model files, csv) into the container
COPY . .

# expose the port streamlit runs on
EXPOSE 8501

# command to run the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]