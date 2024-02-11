import requests
import bs4
from bs4 import BeautifulSoup

import sqlite3
from datetime import datetime


class Biznes_radar_scraper(object):
    """
    Make scraper for gathering data from https://www.biznesradar.pl/
    """
    
    def __init__(self, url, db_name=None):
        """
        url - http web adress from https://www.biznesradar.pl/ of company
        which want to scrap
        db_name - name of db for which we want to connect
        """
        self.url = url
        self.db_name = db_name
        self.flagConnOpen = False
        
    def _parse_data(self, data):
        """
        data transformation function
        """
        return data.replace(' ', '')
    
    def _get_last_page_number(self, soup):
        """
        Get last page of stock data files from website.
        It is used for looping over all pages and later used as stop mark.
        soup - bs4 object used for scrapping data
        """
        return int([x.text for x in soup.find_all("a", class_="pages_pos")][-1])
    
    def _get_data_from_list(self, list):
        """
        helper function, used for taking data from list of scraped data.
        """
        return datetime.strptime(list[0], "%d.%m.%Y"), list[1], list[2], list[3], list[4], list[5], list[6]
    
    def make_new_db(self, cursor, new_db_name):
        """
        make new db with define name and structure
        """
        cursor.execute("""CREATE TABLE {}
                    (
                        date DATETIME PRIMARY KEY,
                        open REAL,
                        MIN REAL,
                        MAX REAL,
                        close REAL,
                        volume REAL,
                        trade REAL
                        )
                    """.format(new_db_name))
        cursor.close()
    
    def conect_db(self, make_new_db=False, table_name=None):
        """
        connect to SQL db, if db not exist make new one
        """
        if not self.db_name:
            self.db_name = str(input("Insert name of your db:  "))+".db"
        
        if not make_new_db:
            self.db = sqlite3.connect(self.db_name)
            if self.db_name[-2]!=".db":
                self.db_name =self.db_name+".db" 
                print("Database name must have an suffix .db")
                print("Adding the sufix")
                print(f"Actual db_name: {self.db_name}")

            self.flagConnOpen = True
            print(f"Succesfuly coneccted to: {self.db_name}.db")

        else:
            self.new_db_name = self.db_name
            self.cursor = sqlite3.connect(self.new_db_name).cursor()
            self.make_new_db(self.cursor , table_name)
            print(f"New database {self.new_db_name} have been created!")
            self.db = sqlite3.connect(self.new_db_name)
            self.flagConnOpen = True
            print(f"Succesfuly coneccted to: {self.new_db_name}")
            
        return self.db, self.flagConnOpen
    
    def _scrap(self, url, db, table_name, conection_flag):
        """
        Scrap data from website url
        """
        self.page = requests.get(url)
        self.soup = BeautifulSoup(self.page.content, "html.parser")
        self.i = 1
        self.cursor = db.cursor()
        if conection_flag:
            try:
                while self.i < self._get_last_page_number(self.soup):
                    if self.i==1:
                        pass
                    else:
                        self.new_url = url+","+str(self.i)
                        self.page = requests.get(self.new_url)
                        self.soup = BeautifulSoup(self.page.content, "html.parser")
                    self.table = self.soup.find("table", class_="qTableFull")
                    self.rows = self.table.find_all("tr")    
                    
                    for self.row in self.rows:
                        self.cols = self.row.find_all('td')
                        self.cols = [ele.text.strip() for ele in self.cols]
                        if bool(self.cols):
                                #data.append([parse_data(ele) for ele in cols if ele])
                                self.data = [self._parse_data(ele) for ele in self.cols if ele]
                                self.date, self.open_, self.min_ \
                                ,self.max_, self.close, self.volume, self.trade = self._get_data_from_list(self.data)
                                #print(f"DATE: {self.date}")
                                self.cursor.execute(
                                    "INSERT INTO {} VALUES (?,?,?,?,?,?,?)".format(table_name),
                                    (self.date, self.open_, self.min_, self.max_, self.close, self.volume, self.trade)
                                    )  
                        else:
                            continue
                    
                        db.commit()
                        self.i+=1
            except sqlite3.IntegrityError:
                print("=====Database has been updated!!=====\n")
                print(f"=====Last recorded date is: {self.date}=====")
            
            db.close()
        
        else:
            print("Database not connected!!!")
    
    def get_stock_data(self, table_name, make_new_db=False):
        if not make_new_db:
            self.db_conected, self.conection_flag = self.conect_db()
            
        else:
            self.db_conected, self.conection_flag = self.conect_db(True, table_name)
            
        self._scrap(self.url, self.db_conected, table_name, self.conection_flag)
        print("========Scrapping Complete!========")
        print(f"========Data scrapped to {table_name} in {self.db_name} database========")
            
            
            
url = "https://www.biznesradar.pl/notowania-historyczne/LOTOS"   
db_name = "test"               
scraper = Biznes_radar_scraper(url)
scraper.get_stock_data("test_table", make_new_db=True)
        
        
    
    

