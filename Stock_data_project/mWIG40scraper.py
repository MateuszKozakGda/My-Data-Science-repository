import scrapy


class Mwig40scraperSpider(scrapy.Spider):
    name = 'mWIG40scraper'
    #allowed_domains = ['https://www.biznesradar.pl/']
    start_urls = start_urls = ["https://www.biznesradar.pl/gielda/indeks:mWIG40"]
    def parse(self, response):
        for link in response.css("td").css("a.s_tt::attr(href)"):
            #print(link.get())
            yield response.follow(link.get(), callback=self.parse_link)
    
    def parse_link(self, response):
        link = response.css("li[data-key='208'] a::attr(href)")
        #print(link.get())
        yield response.follow(link.get(), callback=self.parse_historian)
    
    def parse_historian(self, response):
        name = response.css("h2::text").get()
        #print(name)
        yield {
               "ticker" : name
               }
        #i==0
        #table = response.css("table.qTableFull tr")
        #for row in table:
        #    print(row.css("td::text").get())