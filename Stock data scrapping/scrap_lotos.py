import bs4
import requests
from bs4 import BeautifulSoup
import numpy as np
import sqlite3
from sys import argv
from datetime import datetime


url = "https://www.biznesradar.pl/notowania-historyczne/LOTOS"

page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")


def parse_data(data):
    return data.replace(' ', '')

def get_last_page_number(soup):
    return int([x.text for x in soup.find_all("a", class_="pages_pos")][-1])

def get_data_from_list(list):
    return datetime.strptime(list[0], "%d.%m.%Y"), list[1], list[2], list[3], list[4], list[5], list[6]


db = sqlite3.connect("stock.db")
cursor = db.cursor()

### Create db when using setup comand in terminal
if len(argv)>1 and argv[1] == "setup":
    cursor.execute("""CREATE TABLE lotos_stock
                   (
                       date DATETIME PRIMARY KEY,
                       open REAL,
                       MIN REAL,
                       MAX REAL,
                       close REAL,
                       volume REAL,
                       trade REAL
                       )
                   """)
    quit()


def scrap(soup):
    i=1
    try:
        while i<get_last_page_number(soup):
            if i==1:
                table =  soup.find("table", class_="qTableFull")
                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    if bool(cols):
                        #data.append([parse_data(ele) for ele in cols if ele])
                        data = [parse_data(ele) for ele in cols if ele]
                        date, open_, min_, max_, close, volume, trade = get_data_from_list(data)
                        print(f"DATE: {date}")
                        cursor.execute(
                            "INSERT INTO lotos_stock VALUES (?,?,?,?,?,?,?)",
                            (date, open_, min_, max_, close, volume, trade)
                            )  
                    else:
                        continue
                    
            else:
                page = requests.get(f"https://www.biznesradar.pl/notowania-historyczne/LOTOS,{i}")
                soup = BeautifulSoup(page.content, "html.parser")
                table =  soup.find("table", class_="qTableFull")
                rows = table.find_all("tr")
                for row in rows:
                    cols = row.find_all('td')
                    cols = [ele.text.strip() for ele in cols]
                    if bool(cols):
                        #data.append([parse_data(ele) for ele in cols if ele])
                        data = [parse_data(ele) for ele in cols if ele]
                        date, open_, min_, max_, close, volume, trade = get_data_from_list(data)
                        cursor.execute(
                            "INSERT INTO lotos_stock VALUES (?,?,?,?,?,?,?)",
                            (date, open_, min_, max_, close, volume, trade)
                            )

                    else:
                        continue
            db.commit() 
            i+=1
    except sqlite3.IntegrityError:
        print("=====Database has been updated!!=====\n")
        print(f"=====Last recorded date is: {date}=====")

#data = []
if len(argv)>1 and argv[1] == "scrap":
    scrap(soup)
    
db.close()