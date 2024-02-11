import scrapy
from scrapy.exceptions import CloseSpider
from WIG40.items import Wig40Item
from scrapy.loader import ItemLoader
from datetime import datetime

class TableSpider(scrapy.Spider):
    name = 'TABLE'
    start_urls = ['https://www.biznesradar.pl/notowania-historyczne/AMBRA']

    def parse(self, response):
        name = response.css("h2::text").get()
        for row in response.xpath('//*[@class="qTableFull"]//tr') :
            a = str(row.xpath('td[1]//text()').extract_first())
            if a=="None":
                continue
            elif int(datetime.strptime(a, '%d.%m.%Y').year) > 2020:
                l = ItemLoader(item=Wig40Item(), selector = row)
                l.add_value("ticker", name)
                l.add_xpath("date", "td[1]")
                l.add_xpath("open", "td[2]")
                l.add_xpath("MAX", "td[3]")
                l.add_xpath("MIN", "td[4]")
                l.add_xpath("close", "td[5]")
                l.add_xpath("volume", "td[6]")
                l.add_xpath("trade", "td[7]")
                
                yield l.load_item()
                
            else:
                raise CloseSpider("======JOB FINISHED=====")
                    
        next_page = response.css("a.pages_right").attrib["href"]
        if next_page is not None:
            yield response.follow(next_page, callback=self.parse)

