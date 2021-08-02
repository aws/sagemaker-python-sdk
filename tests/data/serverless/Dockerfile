FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

COPY model.py .
RUN python model.py

COPY app.py .
COPY classes.txt .

CMD ["app.handler"]
