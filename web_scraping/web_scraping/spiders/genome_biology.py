import scrapy

from web_scraping.utils import parse_article

class GenomeBiologySpider(scrapy.Spider):
    name = 'genome_biology'
    start_urls = ['https://genomebiology.biomedcentral.com/articles']

    BASE_URL = 'https://genomebiology.biomedcentral.com'

    def parse(self, response):
        for article in response.css('h3.c-listing__title a::attr(href)').getall():
            absolute_url = response.urljoin(article)
            yield scrapy.Request(absolute_url, callback=parse_article)

        next_page = response.css('a[rel="next"]::attr(href)').get()
        if next_page:
            absolute_next_page = response.urljoin(next_page)
            self.logger.info(f"Following next page: {absolute_next_page}")
            yield scrapy.Request(absolute_next_page, callback=self.parse)
