FROM python:3.9.13

WORKDIR /app

COPY ./src .

RUN pip install matplotlib seaborn scikit-learn pandas numpy argparse plotly sys

CMD ["bash"]
