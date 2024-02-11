import scrapy


class mWIG40spider(scrapy.Spider):
    name = "mWIG40"
    start_urls = ["https://www.biznesradar.pl/gielda/indeks:mWIG40"]
    
    def parse(self, response):
        for item in response.css("td"):
            if item.css('a::text'):
                yield {
                    "ticker_name" : item.css('a::text').get()
                        }
            else:
                continue
      