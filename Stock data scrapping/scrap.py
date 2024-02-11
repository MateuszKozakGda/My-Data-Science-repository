import bs4
import requests
from bs4 import BeautifulSoup
from fastnumbers import isfloat 
from fastnumbers import fast_float


url = "https://www.gpw.pl/spolka?isin=PLPKN0000018"


def parse_price(url):
    page_respone = requests.get(url, timeout=240)
    soup = BeautifulSoup(page_respone.text, "xml")
    value = soup.find_all("div", {"class" : "PaL header text-right text-left-xs"})[0].find("span").text
    return value

def get_children(html_content):
    return [item for item in html_content.children if type(item)==bs4.element.Tag or len(str(item).replace("\n","").strip())>0]

def remove_multiple_spaces(string):
    if type(string)==str:
        return ' '.join(string.split())
    return string

def get_table_simple(table,is_table_tag=True):
    elems = table.find_all('tr') if is_table_tag else get_children(table)
    table_data = list()
    for row in elems:
        row_data = list()
        row_elems = get_children(row)
        for elem in row_elems:
            text = elem.text.strip().replace("\n","")
            text = remove_multiple_spaces(text)
            if len(text)==0:
                continue
            row_data.append(text)
        table_data.append(row_data)
    return table_data

page_respone = requests.get(url, timeout=240)
soup = BeautifulSoup(page_respone.text, "xml")

print(get_table_simple(soup.find_all("div", {"class" : "max_min"})[0], is_table_tag=False))
      

"""def parse_MIN_MAX(url):
    page_respone = requests.get(url, timeout=240)
    soup = BeautifulSoup(page_respone.text, "xml")
    value = soup.find_all("div", {"class" : "PaL header text-right text-left-xs"})[0].find("span").text

#while True:
    #print("Obecna cena akcji to: ", str(parse_price(url)))"""
