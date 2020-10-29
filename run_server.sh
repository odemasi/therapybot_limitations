# gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:6776 server:app

# navigate to: https://language.cs.ucdavis.edu/covidbot/user/

gunicorn --worker-class eventlet -w 1 -b 127.0.0.1:6776 server:app


