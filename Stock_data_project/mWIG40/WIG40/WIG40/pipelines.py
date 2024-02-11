# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import sqlite3

class Wig40Pipeline:
    def __init__(self):
        self.con = sqlite3.connect("Wig40.db")
        self.cur = self.con.cursor()
        self.create_table()
    
    def create_table(self):
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS wig40
            (
                date DATETIME,
                ticker VARCHAR,
                open REAL,
                MAX REAL,
                MIN REAL,
                close REAL,
                volume REAL,
                trade REAL,
                PRIMARY KEY(date, ticker)
                )
                """
                         )
        
    def process_item(self, item, spider):
        self.cur.execute(
            """INSERT OR IGNORE INTO wig40 VALUES(?,?,?,?,?,?,?,?)""",
            (item["date"], 
            item["ticker"], 
            item["open"], 
            item["MAX"], 
            item["MIN"], 
            item["close"], 
            item["volume"], 
            item["trade"])
        )
        self.con.commit()
        return item
    
class Wig20Pipeline:
    def __init__(self):
        self.con = sqlite3.connect("Wig40.db")
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS wig20
            (
                date DATETIME,
                ticker VARCHAR,
                open REAL,
                MAX REAL,
                MIN REAL,
                close REAL,
                volume REAL,
                trade REAL,
                PRIMARY KEY(date, ticker)
                )
                """
                         )
        
    def process_item(self, item, spider):
        self.cur.execute(
            """INSERT OR IGNORE INTO wig20 VALUES(?,?,?,?,?,?,?,?)""",
            (item["date"], 
            item["ticker"], 
            item["open"], 
            item["MAX"], 
            item["MIN"], 
            item["close"], 
            item["volume"], 
            item["trade"])
        )
        self.con.commit()
        return item
    
class Wig80Pipeline:
    def __init__(self):
        self.con = sqlite3.connect("Wig40.db")
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS wig80
            (
                date DATETIME,
                ticker VARCHAR,
                open REAL,
                MAX REAL,
                MIN REAL,
                close REAL,
                volume REAL,
                trade REAL,
                PRIMARY KEY(date, ticker)
                )
                """
                         )
        
    def process_item(self, item, spider):
        self.cur.execute(
            """INSERT OR IGNORE INTO wig80 VALUES(?,?,?,?,?,?,?,?)""",
            (item["date"], 
            item["ticker"], 
            item["open"], 
            item["MAX"], 
            item["MIN"], 
            item["close"], 
            item["volume"], 
            item["trade"])
        )
        self.con.commit()
        return item   
    
    
class Wig20Pipeline:
    def __init__(self):
        self.con = sqlite3.connect("Wig40.db")
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS wig20
            (
                date DATETIME,
                ticker VARCHAR,
                open REAL,
                MAX REAL,
                MIN REAL,
                close REAL,
                volume REAL,
                trade REAL,
                PRIMARY KEY(date, ticker)
                )
                """
                         )
        
    def process_item(self, item, spider):
        self.cur.execute(
            """INSERT OR IGNORE INTO wig20 VALUES(?,?,?,?,?,?,?,?)""",
            (item["date"], 
            item["ticker"], 
            item["open"], 
            item["MAX"], 
            item["MIN"], 
            item["close"], 
            item["volume"], 
            item["trade"])
        )
        self.con.commit()
        return item
    
class WIGPipeline:
    def __init__(self):
        self.con = sqlite3.connect("WIG.db")
        self.cur = self.con.cursor()
        self.create_table()

    def create_table(self):
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS WIG
            (
                date DATETIME,
                ticker VARCHAR,
                open REAL,
                MAX REAL,
                MIN REAL,
                close REAL,
                volume REAL,
                trade REAL,
                PRIMARY KEY(date, ticker)
                )
                """
                         )
        
    def process_item(self, item, spider):
        self.cur.execute(
            """INSERT OR IGNORE INTO WIG VALUES(?,?,?,?,?,?,?,?)""",
            (item["date"], 
            item["ticker"], 
            item["open"], 
            item["MAX"], 
            item["MIN"], 
            item["close"], 
            item["volume"], 
            item["trade"])
        )
        self.con.commit()
        return item 
    
    
    
    
    
    




