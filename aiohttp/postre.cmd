docker run --rm -it -p 5432:5432 postgres:10
psql -U postgres -h localhost
CREATE DATABASE aiohttpdemo_polls;
CREATE USER aiohttpdemo_user WITH PASSWORD 'aiohttpdemo_pass';
GRANT ALL PRIVILEGES ON DATABASE aiohttpdemo_polls TO aiohttpdemo_user;
