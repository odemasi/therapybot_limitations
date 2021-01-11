# gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:6776 server:app

# navigate to: https://language.cs.ucdavis.edu/covidbot/user/

CUDA_VISIBLE_DEVICES=2,7,3 gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:6776 server:app




