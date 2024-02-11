import scrapy
from scrapy.exceptions import CloseSpider
from WIG40.items import Wig40Item, Wig20Item, Wig80Item, WIGitem
from scrapy.loader import ItemLoader
from datetime import datetime

#from scrapy.crawler import CrawlerProcess

today_date = datetime.strptime(datetime.now().strftime('%d.%m.%Y'), '%d.%m.%Y')

class Mwig40spiderSpider(scrapy.Spider):
    name = 'mWIG40spider'
    custom_settings = { 'ITEM_PIPELINES':
        {'WIG40.pipelines.Wig40pipeline': 401}
    }

    def start_requests(self):
        start_urls = [
                    'https://www.biznesradar.pl/gielda/indeks:mWIG40'
                  ]
        for i in start_urls:
            yield scrapy.Request(url=i,callback=self.parse)

    def parse(self, response):
        for link in response.css("td").css("a.s_tt::attr(href)"):
            yield response.follow(link.get(), callback=self.parse_link)
    
    def parse_link(self, response):
        link = response.css("li[data-key='208'] a::attr(href)")
        yield response.follow(link.get(), callback=self.parse_historian)
    
    def parse_historian(self, response):
        name = response.css("h2::text").get()
        for row in response.xpath('//*[@class="qTableFull"]//tr') :
            a = str(row.xpath('td[1]//text()').extract_first())
            if a=="None":
                continue
            elif int(datetime.strptime(a, '%d.%m.%Y').year) > self.year:
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
            yield response.follow(next_page, callback=self.parse_historian)
            

            
class Mwig80spiderSpider(scrapy.Spider):
    name = 'WIG80spider'
    custom_settings = { 'ITEM_PIPELINES':
        {'WIG40.pipelines.WIig80ipeline': 401}
    }
    
    def start_requests(self):
        start_urls = [
                    'https://www.biznesradar.pl/gielda/indeks:sWIG80'
                  ]
        for i in start_urls:
            yield scrapy.Request(url=i,callback=self.parse)

    def parse(self, response):
        for link in response.css("td").css("a.s_tt::attr(href)"):
            yield response.follow(link.get(), callback=self.parse_link)
    
    def parse_link(self, response):
        link = response.css("li[data-key='208'] a::attr(href)")
        yield response.follow(link.get(), callback=self.parse_historian)
    
    def parse_historian(self, response):
        name = response.css("h2::text").get()
        for row in response.xpath('//*[@class="qTableFull"]//tr') :
            a = str(row.xpath('td[1]//text()').extract_first())
            if a=="None":
                continue
            elif int(datetime.strptime(a, '%d.%m.%Y').year) > self.year:
                l = ItemLoader(item=Wig80Item(), selector = row)
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
            yield response.follow(next_page, callback=self.parse_historian)
            

class Mwig20spiderSpider(scrapy.Spider):
    name = 'WIG20spider'
    custom_settings = { 'ITEM_PIPELINES':
        {'WIG40.pipelines.WIig20ipeline': 401}
    }

    def start_requests(self):
        start_urls = [
                    'https://www.biznesradar.pl/gielda/indeks:WIG20'
              ]
        for i in start_urls:
            yield scrapy.Request(url=i,callback=self.parse)

    def parse(self, response):
        for link in response.css("td").css("a.s_tt::attr(href)"):
            yield response.follow(link.get(), callback=self.parse_link)
    
    def parse_link(self, response):
        link = response.css("li[data-key='208'] a::attr(href)")
        yield response.follow(link.get(), callback=self.parse_historian)
    
    def parse_historian(self, response):
        name = response.css("h2::text").get()
        for row in response.xpath('//*[@class="qTableFull"]//tr') :
            a = str(row.xpath('td[1]//text()').extract_first())
            if a=="None":
                continue
            elif int(datetime.strptime(a, '%d.%m.%Y').year) > self.year:
                l = ItemLoader(item=Wig20Item(), selector = row)
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
            yield response.follow(next_page, callback=self.parse_historian)
            
class WIGspiderSpider(scrapy.Spider):
    name = 'WIGspider'
    custom_settings = { 'ITEM_PIPELINES':
        {'WIG40.pipelines.WIGPipeline': 401}
    }
    
    def start_requests(self):
        start_urls = [
                    'https://www.biznesradar.pl/gielda/indeks:WIG'
                  ]
        for i in start_urls:
            yield scrapy.Request(url=i,callback=self.parse, dont_filter=True)

    def parse(self, response):
        for link in response.css("td").css("a.s_tt::attr(href)"):
            yield response.follow(link.get(), callback=self.parse_link)
    
    def parse_link(self, response):
        link = response.css("li[data-key='208'] a::attr(href)")
        yield response.follow(link.get(), callback=self.parse_historian)
    
    def parse_historian(self, response):
        name = response.css("h2::text").get()
        try:
            for row in response.xpath('//*[@class="qTableFull"]//tr') :
                a = str(row.xpath('td[1]//text()').extract_first())
                if a=="None":
                    continue

                elif int(datetime.strptime(a, '%d.%m.%Y').year) > int(self.year):
                    l = ItemLoader(item=WIGitem(), selector = row)
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
                    #raise CloseSpider("======JOB FINISHED=====")
                    print(f"======JOB FINISHED on date: {datetime.strptime(a, '%d.%m.%Y')}=====")
                    break
                
                next_page = response.css("a.pages_right").attrib["href"]
                try:
                    if next_page is not None:
                        yield response.follow(next_page, callback=self.parse_historian)
                except UnboundLocalError:
                    continue
            
        except KeyError:
            pass


#process = CrawlerProcess()
#process.crawl(Mwig40spiderSpider)
#process.crawl(Mwig20spiderSpider)
#process.crawl(Mwig80spiderSpider)
#process.start()
