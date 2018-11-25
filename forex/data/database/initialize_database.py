import sqlite3

con = sqlite3.connect("FOREX")

con.executescript("schema.sql")

con.execute("SHOW TABLES;")