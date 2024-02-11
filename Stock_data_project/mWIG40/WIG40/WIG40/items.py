# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.loader import ItemLoader
from itemloaders.processors import TakeFirst, MapCompose
from w3lib.html import remove_tags

def remove_blank(value):
    return value.replace(" ", "").strip()

def remove_letters_error(word):
    return word.replace("\u00d3", "O").replace("\u0141", "L")

def parse_date(date):
    return date.replace(".", "-")

class Wig40Item(scrapy.Item):
    ticker = scrapy.Field (input_processor = MapCompose(remove_letters_error, remove_tags),output_processor=TakeFirst())
    date = scrapy.Field (input_processor = MapCompose(parse_date, remove_tags),output_processor=TakeFirst())
    open = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MAX = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MIN = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    close = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    volume = scrapy.Field(input_processor = MapCompose(remove_tags, remove_blank), output_processor=TakeFirst())
    trade =  scrapy.Field(input_processor = MapCompose(remove_tags,remove_blank), output_processor=TakeFirst())
    
class Wig20Item(scrapy.Item):
    ticker = scrapy.Field (input_processor = MapCompose(remove_letters_error, remove_tags),output_processor=TakeFirst())
    date = scrapy.Field (input_processor = MapCompose(parse_date, remove_tags),output_processor=TakeFirst())
    open = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MAX = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MIN = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    close = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    volume = scrapy.Field(input_processor = MapCompose(remove_tags, remove_blank), output_processor=TakeFirst())
    trade =  scrapy.Field(input_processor = MapCompose(remove_tags,remove_blank), output_processor=TakeFirst())
    
class Wig80Item(scrapy.Item):
    ticker = scrapy.Field (input_processor = MapCompose(remove_letters_error, remove_tags),output_processor=TakeFirst())
    date = scrapy.Field (input_processor = MapCompose(parse_date, remove_tags),output_processor=TakeFirst())
    open = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MAX = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MIN = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    close = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    volume = scrapy.Field(input_processor = MapCompose(remove_tags, remove_blank), output_processor=TakeFirst())
    trade =  scrapy.Field(input_processor = MapCompose(remove_tags,remove_blank), output_processor=TakeFirst())
    
class WIGitem(scrapy.Item):
    ticker = scrapy.Field (input_processor = MapCompose(remove_letters_error, remove_tags),output_processor=TakeFirst())
    date = scrapy.Field (input_processor = MapCompose(parse_date, remove_tags),output_processor=TakeFirst())
    open = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MAX = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    MIN = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    close = scrapy.Field (input_processor = MapCompose(remove_tags),output_processor=TakeFirst())
    volume = scrapy.Field(input_processor = MapCompose(remove_tags, remove_blank), output_processor=TakeFirst())
    trade =  scrapy.Field(input_processor = MapCompose(remove_tags,remove_blank), output_processor=TakeFirst())
    
