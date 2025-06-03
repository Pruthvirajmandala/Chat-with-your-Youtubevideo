import os
from flask import Flask, render_template, request, Response, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from http import HTTPStatus
from utils import load, generate
from models import db, User

app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'
app.secret_key = os.getenv("SECRET_KEY")
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('auth.html', mode='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return render_template('auth.html', mode='register')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists')
            return render_template('auth.html', mode='register')
        
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        login_user(user)
        return redirect(url_for('index'))
    
    return render_template('auth.html', mode='register')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET'])
@login_required
def index():
    return render_template('index.html')

@app.route('/load', methods=['POST'])
@login_required
def load_controller():
    status, data = load(request.form.get('url'))
    if status == 'error':
        return Response(data, status=HTTPStatus.BAD_REQUEST)
    response = Response()
    user_namespace = f"{current_user.id}_{data}"
    response.set_cookie('name_space', user_namespace)
    return response

@app.route('/generate', methods=['POST'])
@login_required
def generate_controller():
    prompt = request.form.get('prompt')
    model = request.form.get('model')
    name_space = request.cookies.get('name_space')
    if not name_space or not name_space.startswith(f"{current_user.id}_"):
        return Response("Please load a video first", status=HTTPStatus.BAD_REQUEST)
    return Response(generate(model, name_space, prompt))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='127.0.0.1', port=5001, threaded=True)
