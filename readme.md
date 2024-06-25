# Devolve: A Text-to-SQL System

Devolve is a sophisticated text-to-SQL system designed to translate natural language queries into SQL commands. This project leverages the CodeLlama-34B model, fine-tuned on the Spider SQL dataset, to achieve high accuracy and efficiency in generating SQL queries.

## Project Overview

Devolve uses advanced machine learning techniques to convert natural language questions into SQL queries. It is designed to support various types of databases and includes a demonstration web application.

## Features

- **Natural Language to SQL**: Converts plain English questions into SQL queries.
- **Supports Multiple Databases**: Interacts with SQLite, MySQL, PostgreSQL, Oracle, SQL Server, and MongoDB.
- **Web Application**: A Flask-based web interface for demonstrating the model's capabilities with SQLite databases.
- **Command-Line Interface**: A CLI tool for interacting with various database types beyond SQLite.


## Files
To fine-tune the model, run train.py:

``` python train.py ```

To evaluate the model's performance, run test.py:

   ``` python test.py```

5. Interaction with the Model

    For SQLite support and demonstration using the web application, run app.py:

```python app.py```

For interacting with various types of databases beyond SQLite using the command-line interface, run Interact.py:'

```python Interact.py```
## Examples

### Converting Natural Language to SQL

Input:
```
Show me all the employees in the Sales department.
```

Output:
```sql
SELECT * FROM employees WHERE department = 'Sales';
```

### Interpreting Results

Input SQL:
```sql
SELECT * FROM employees WHERE department = 'Sales';
```

Output:
```
There are 10 employees in the Sales department.
```


## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Inspiration from various text-to-SQL research projects.
- Utilizes the Spider SQL dataset for training.
- Special thanks to all contributors and supporters.

## Contact

For questions or support,contact:

- **Name**: El Mouaquit Nizar
- **Email**: nizarelmouaquit@protonmail.com