release: python -c "from application import app, db; app.app_context().push(); db.create_all()"
web: gunicorn --workers 4 --threads 2 application:app
