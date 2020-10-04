# dev_proj1
동현,준석,인광 개발 프로젝트

# 파일 설명
```app.py``` : web app 실행을 위한 file

```config.py```: 파일 저장 경로 등의 설정 파일

```static folder``` : js, css, images 등 정적 웹 서버를 위한 file

```python_code folder``` : python 코드 기반 함수 및 모델이 포함된 폴더

```templates folder``` : html file
Flask가 templates 폴더의 경로를 자동으로 인식하는 듯 함

```Pipfile, Pipfile.lock ``` : Pipfile 파일과 Pipfile.lock 파일을 내려받은 후에 pipenv install 커맨드 하나로 모든 패키지를 한 방에 설치할 수 있음

# Jinja2란
* Python 웹 프레임워크인 Flask에 내장되어 있는 Template 엔진
* html 파일에 변수를 넣어서 template을 렌더링하여 보여줌
* templates/profile.html 에서 코드 변경 필요

```html
<script type="text/javascript" src="{{ url_for('static', filename='jquery/jquery-3.5.1.min.js')}}"></script>
<link rel="stylesheet" href="{{url_for('static',filename='styles.css')}}">
```
ex) ```{{ variable }}```을 통해 파이썬 variable를 연동시킴. \
즉 url_for('static',filename='styles.css')의 파이썬 함수를 통해 'static' 폴더의 styles.css를 불러오는 형식
